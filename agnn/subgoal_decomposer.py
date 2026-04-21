"""
Tier-2 Subgoal Decomposer — LLM-driven with smart fallback

The LLM decides the full DAG shape for each task:
  - How many subgoals (1–6)
  - What each one is called and does
  - What phase type it is (not restricted to 4 hardcoded types)
  - Which subgoals depend on which (parallelism emerges naturally)

Fallback: if LLM fails or returns junk, a lightweight complexity analyser
produces a minimal correct DAG (1–3 subgoals) without forcing unnecessary phases.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Set

from .llm_client import chat_completion


# ── canonical phase types + aliases ──────────────────────────────────────────

# These are the values the orchestrator knows how to configure thresholds /
# turn budgets / word budgets for. Unknown types fall back to "draft".
CANONICAL_PHASES: Set[str] = {"research", "analysis", "draft", "review"}

_PHASE_ALIASES: Dict[str, str] = {
    "gather": "research", "investigate": "research", "explore": "research",
    "collect": "research", "scout": "research", "study": "research",
    "survey": "research", "discover": "research", "find": "research",
    "assess": "analysis", "evaluate": "analysis", "plan": "analysis",
    "planning": "analysis", "strategize": "analysis", "synthesize": "analysis",
    "design": "analysis", "examine": "analysis", "compare": "analysis",
    "write": "draft", "create": "draft", "develop": "draft",
    "build": "draft", "implement": "draft", "execution": "draft",
    "produce": "draft", "generate": "draft", "compose": "draft",
    "check": "review", "validate": "review", "validation": "review",
    "audit": "review", "test": "review", "finalize": "review",
    "verify": "review", "qa": "review", "proofread": "review",
    "critique": "review", "refine": "review",
}


def _canonicalize_phase(raw: str) -> str:
    """Map any LLM-produced phase string to a canonical type."""
    s = raw.strip().lower()
    if s in CANONICAL_PHASES:
        return s
    if s in _PHASE_ALIASES:
        return _PHASE_ALIASES[s]
    # partial match
    for alias, canon in _PHASE_ALIASES.items():
        if alias in s or s in alias:
            return canon
    return "draft"  # safe default


# ── data model ───────────────────────────────────────────────────────────────

@dataclass
class Subgoal:
    id: int
    name: str
    description: str
    completion_criteria: str
    estimated_turns: int
    phase_type: str          # always a canonical phase after normalisation
    dependencies: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


# ── decomposer ────────────────────────────────────────────────────────────────

class SubgoalDecomposer:
    """
    LLM-first decomposer.

    Priority order:
      1. LLM JSON decomposition  (full flexibility)
      2. Complexity-aware minimal fallback  (no LLM needed)
    """

    # System prompt — tight, structured, works well with 1–3B local models
    _SYSTEM = """\
You are a task planner. Given a task, output a JSON array describing the \
minimal set of steps needed to complete it.

Rules:
- Use ONLY as many steps as the task genuinely needs (1 to 6).
- Simple tasks (summarise, list, explain, write X) need just 1 step.
- Only add a research step if external knowledge must be gathered first.
- Only add a review step if quality-checking is important.
- "deps" lists the IDs of steps that must finish before this one starts.
  Steps with no deps run in parallel with each other.
- "phase" must be ONE of: research, analysis, draft, review
- "turns" is an integer estimate of how many agent turns this step needs (1–8).

Output ONLY valid JSON, no explanation. Example:
[
  {"id":1,"name":"Gather Data","description":"Collect market data","phase":"research","deps":[],"turns":4},
  {"id":2,"name":"Write Report","description":"Draft the report","phase":"draft","deps":[1],"turns":5}
]"""

    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.base_url = base_url
        self.model    = model
        self.timeout  = timeout

    # ── public entry point ────────────────────────────────────────────────────

    def decompose(self, user_prompt: str, roles: list = None) -> List[Subgoal]:
        """
        Decompose *user_prompt* into a validated Subgoal DAG.

        If *roles* are supplied (2+), the LLM is seeded with role names so it
        can create role-specific parallel branches naturally (no forced template).
        """
        role_hint = ""
        if roles and len(roles) >= 2:
            names = [getattr(r, "name", str(r)) for r in roles]
            role_hint = (
                f"\n\nAvailable specialist roles: {', '.join(names)}. "
                "You may create parallel research branches scoped to each role "
                "where that adds value — but only if the task genuinely benefits."
            )

        print("[Subgoal] Asking LLM to plan the DAG...")
        try:
            subgoals = self._llm_decompose(user_prompt + role_hint)
            if subgoals:
                print(f"[Subgoal] LLM produced {len(subgoals)} subgoal(s):")
                for sg in subgoals:
                    deps = f" ← {sg.dependencies}" if sg.dependencies else " (parallel)"
                    print(f"          [{sg.id}] {sg.name} ({sg.phase_type}){deps}")
                return subgoals
        except Exception as exc:
            print(f"[Subgoal] LLM planning failed: {exc}")

        # fallback
        subgoals = self._minimal_fallback(user_prompt)
        print(f"[Subgoal] Fallback: {len(subgoals)} subgoal(s)")
        for sg in subgoals:
            print(f"          [{sg.id}] {sg.name} ({sg.phase_type})")
        return subgoals

    # ── LLM decomposition ─────────────────────────────────────────────────────

    def _llm_decompose(self, prompt: str) -> List[Subgoal]:
        resp = chat_completion(
            system_prompt=self._SYSTEM,
            user_prompt=f"Task: {prompt}",
            model=self.model,
            base_url=self.base_url,
            timeout=self.timeout,
            max_tokens=400,
            temperature=0.2,
        )
        raw = resp.text.strip()
        return self._parse_and_validate(raw)

    # ── parsing & validation ─────────────────────────────────────────────────

    # Generic names a weak model returns when it doesn't understand the task
    _GENERIC_NAMES = {
        "gather data", "write report", "step 1", "step 2", "step 3",
        "task 1", "task 2", "introduction", "conclusion", "overview",
        "data collection", "output", "result", "response",
    }

    def _is_generic_response(self, subgoals: List[Subgoal]) -> bool:
        """Return True if the LLM produced boilerplate names unrelated to the task."""
        if not subgoals:
            return True
        generic_count = sum(
            1 for sg in subgoals
            if sg.name.lower().strip() in self._GENERIC_NAMES
        )
        return generic_count >= len(subgoals) * 0.6  # 60%+ generic → reject

    def _infer_deps(self, subgoals: List[Subgoal]) -> List[Subgoal]:
        """
        When the LLM omits dependency info, infer a sensible ordering:
        - All pure-research subgoals with no deps stay parallel (dep=[])
        - analysis / draft / review depend on every research subgoal that
          was created without deps (i.e. they are upstream data)
        - review additionally depends on draft

        This preserves parallel research branches while enforcing correct
        sequential ordering for downstream phases.
        """
        all_empty = all(not sg.dependencies for sg in subgoals)
        if not all_empty:
            return subgoals  # LLM provided deps — trust them

        research_ids = [sg.id for sg in subgoals if sg.phase_type == "research"]
        draft_ids    = [sg.id for sg in subgoals if sg.phase_type == "draft"]

        for sg in subgoals:
            if sg.phase_type == "research":
                sg.dependencies = []  # parallel
            elif sg.phase_type in ("analysis", "draft"):
                sg.dependencies = [r for r in research_ids if r != sg.id]
            elif sg.phase_type == "review":
                sg.dependencies = [
                    d for d in (research_ids + draft_ids) if d != sg.id
                ]
            else:
                # unknown phase: depend on everything before it
                prior = [s.id for s in subgoals if s.id < sg.id]
                sg.dependencies = prior[-1:] if prior else []

        return subgoals

    def _parse_and_validate(self, raw: str) -> List[Subgoal]:
        """Extract JSON from LLM output and convert to validated Subgoal list."""
        # strip markdown fences if present
        raw = re.sub(r"```[a-zA-Z]*", "", raw, flags=re.IGNORECASE).strip().rstrip("`").strip()

        # find first [...] block
        m = re.search(r"\[\s*\{.*?\}\s*\]", raw, re.DOTALL)
        if not m:
            # Maybe they wrapped it in {"subgoals": [...]}
            m = re.search(r'"(?:subgoals|roles|tasks)"\s*:\s*(\[\s*\{.*?\}\s*\])', raw, re.DOTALL)
            if not m:
                raise ValueError("No JSON array found in LLM response")
            data = json.loads(m.group(1))
        else:
            data = json.loads(m.group())

        if not isinstance(data, list) or not data:
            raise ValueError("Empty or non-list JSON")

        # normalise fields
        subgoals: List[Subgoal] = []
        valid_ids: Set[int] = set()

        for item in data[:6]:  # hard cap at 6
            if not isinstance(item, dict):
                continue
            sid   = int(item.get("id", len(subgoals) + 1))
            name  = str(item.get("name", "Step")).strip()[:60]
            desc  = str(item.get("description", name)).strip()[:300]
            phase = _canonicalize_phase(str(item.get("phase", "draft")))
            turns = max(1, min(8, int(item.get("turns", 4))))
            deps_raw = item.get("deps") or item.get("dependencies") or []
            deps  = [int(d) for d in deps_raw if isinstance(d, (int, float))]

            if len(name) < 2:
                continue

            valid_ids.add(sid)
            subgoals.append(Subgoal(
                id=sid,
                name=name,
                description=desc,
                completion_criteria=f"{name} completed",
                estimated_turns=turns,
                phase_type=phase,
                dependencies=[],  # fill after we know all ids
            ))
            subgoals[-1]._raw_deps = deps  # type: ignore[attr-defined]

        if not subgoals:
            raise ValueError("No valid subgoals parsed")

        # Reject if the LLM returned boilerplate generic names
        if self._is_generic_response(subgoals):
            raise ValueError("LLM returned generic/boilerplate subgoal names")

        # resolve raw deps
        for sg in subgoals:
            raw_deps = getattr(sg, "_raw_deps", [])
            sg.dependencies = [d for d in raw_deps if d in valid_ids and d != sg.id]
            try:
                del sg._raw_deps  # type: ignore[attr-defined]
            except AttributeError:
                pass

        # Infer missing deps from phase ordering
        subgoals = self._infer_deps(subgoals)

        # ensure no cycles
        subgoals = self._break_cycles(subgoals)

        # re-sequence IDs to 1..N
        subgoals = self._resequence(subgoals)

        return subgoals

    def _break_cycles(self, subgoals: List[Subgoal]) -> List[Subgoal]:
        """Remove dependency edges that create cycles using DFS."""
        id_map = {sg.id: sg for sg in subgoals}

        def has_path(src: int, dst: int, visited: Set[int]) -> bool:
            if src == dst:
                return True
            if src in visited:
                return False
            visited.add(src)
            sg = id_map.get(src)
            if not sg:
                return False
            return any(has_path(d, dst, visited) for d in sg.dependencies)

        for sg in subgoals:
            safe_deps = []
            for dep in sg.dependencies:
                # adding dep→sg.id: only safe if sg.id cannot already reach dep
                if not has_path(sg.id, dep, set()):
                    safe_deps.append(dep)
            sg.dependencies = safe_deps

        return subgoals

    def _resequence(self, subgoals: List[Subgoal]) -> List[Subgoal]:
        """Re-number IDs 1..N and update dependency references."""
        old_to_new = {sg.id: i + 1 for i, sg in enumerate(subgoals)}
        for sg in subgoals:
            sg.id           = old_to_new[sg.id]
            sg.dependencies = [old_to_new[d] for d in sg.dependencies
                               if d in old_to_new]
        return subgoals

    # ── minimal fallback (no LLM) ─────────────────────────────────────────────

    def _minimal_fallback(self, prompt: str) -> List[Subgoal]:
        """
        Produce the smallest correct DAG based on keyword signals.
        Never forces all 4 phases — uses only what the task needs.
        """
        p = prompt.lower()

        # ── single-step tasks ─────────────────────────────────────────────────
        _simple_verbs = ["summarize", "summarise", "list", "what is", "what are",
                         "define", "explain", "describe", "translate", "convert",
                         "rewrite", "paraphrase", "fix", "correct", "format"]
        if any(p.startswith(v) or f" {v} " in p for v in _simple_verbs):
            phase = "review" if any(w in p for w in ["review", "check", "audit", "proofread"]) else "draft"
            return [Subgoal(1, "Complete Task", prompt[:120],
                            "Task done", 4, phase, [])]

        # ── pure research ─────────────────────────────────────────────────────
        if any(w in p for w in ["research", "find", "gather", "collect", "survey"]) \
                and not any(w in p for w in ["write", "create", "draft", "build", "plan"]):
            return [Subgoal(1, "Research", prompt[:120], "Research complete", 5, "research", [])]

        # ── pure review / audit ───────────────────────────────────────────────
        if any(p.startswith(w) for w in ["review", "audit", "evaluate", "assess", "check"]):
            return [Subgoal(1, "Review & Assess", prompt[:120], "Review complete", 4, "review", [])]

        # ── analysis + output ─────────────────────────────────────────────────
        if any(w in p for w in ["analyze", "analyse", "compare", "evaluate", "assess"]):
            return [
                Subgoal(1, "Analyze", "Investigate and analyze the subject", "Analysis done", 5, "analysis", []),
                Subgoal(2, "Produce Output", "Write the findings", "Output ready", 5, "draft", [1]),
            ]

        # ── complex / planning tasks ──────────────────────────────────────────
        needs_research = any(w in p for w in ["strategy", "plan", "market", "design",
                                               "architecture", "roadmap", "competitive"])
        if needs_research:
            return [
                Subgoal(1, "Research", "Gather relevant information and context", "Research complete", 5, "research", []),
                Subgoal(2, "Draft Output", "Produce the deliverable using research", "Draft complete", 7, "draft", [1]),
            ]

        # ── default: single draft ────────────────────────────────────────────
        return [Subgoal(1, "Complete Task", prompt[:120], "Task done", 6, "draft", [])]

    # ── serialisation helper ──────────────────────────────────────────────────

    def subgoals_to_dict(self, subgoals: List[Subgoal]) -> List[Dict]:
        return [sg.to_dict() for sg in subgoals]
