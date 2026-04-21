"""
Handoff Protocol for AGNN

Defines the structured format for deliverables passed between DAG nodes.
Instead of raw text blobs, each completed workspace produces a structured
HandoffPackage that the downstream workspace can consume cleanly.

This makes the DAG feel like a real pipeline instead of "pass the text forward."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import re
import json

from .llm_client import chat_completion


@dataclass
class HandoffPackage:
    """
    Structured deliverable from a completed workspace.

    Every workspace MUST produce this before marking itself complete.
    Downstream workspaces receive these as structured input instead of raw text.
    """
    subgoal_id: int
    subgoal_name: str
    phase_type: str

    # Core deliverable
    what_was_done: str              # Summary of work completed (2-3 sentences)
    key_deliverable: str            # The actual output content (full text)

    # Downstream guidance
    what_matters_downstream: List[str]   # Key points the next phase should use
    uncertainties: List[str]             # What this phase couldn't resolve
    suggested_focus: str                 # What the next phase should prioritize

    # Quality metadata
    turns_taken: int = 0
    acceptance_rate: float = 0.0     # accepted / (accepted + rejected)
    avg_tis: float = 0.0

    def to_downstream_context(self, max_deliverable_chars: int = 3000) -> str:
        """
        Format this handoff as a context block for a downstream agent's prompt.
        """
        lines = [
            f"[Upstream: {self.subgoal_name} ({self.phase_type})]",
            f"Summary: {self.what_was_done}",
        ]

        if self.what_matters_downstream:
            lines.append("Key points to use:")
            for point in self.what_matters_downstream[:5]:
                lines.append(f"  • {point}")

        if self.uncertainties:
            lines.append("Unresolved (address if possible):")
            for u in self.uncertainties[:3]:
                lines.append(f"  ⚠ {u}")

        if self.suggested_focus:
            lines.append(f"Suggested focus: {self.suggested_focus}")

        # Include truncated deliverable for reference
        deliverable = self.key_deliverable[:max_deliverable_chars]
        if len(self.key_deliverable) > max_deliverable_chars:
            deliverable += "\n[...truncated...]"
        lines.append(f"\n--- Full Output ---\n{deliverable}")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "subgoal_id": self.subgoal_id,
            "subgoal_name": self.subgoal_name,
            "phase_type": self.phase_type,
            "what_was_done": self.what_was_done,
            "what_matters_downstream": self.what_matters_downstream,
            "uncertainties": self.uncertainties,
            "suggested_focus": self.suggested_focus,
            "turns_taken": self.turns_taken,
            "acceptance_rate": round(self.acceptance_rate, 3),
            "avg_tis": round(self.avg_tis, 3),
        }


def create_handoff_package(
    *,
    subgoal_id: int,
    subgoal_name: str,
    phase_type: str,
    raw_deliverable: str,
    accepted_messages: list,
    turns_taken: int,
    rejected_count: int,
    model: str,
    base_url: str,
) -> HandoffPackage:
    """
    Create a structured HandoffPackage from raw workspace output.

    Tries to use the LLM to extract structured fields from the deliverable.
    Falls back to heuristic extraction if the LLM call fails.
    """
    acceptance_rate = turns_taken / max(turns_taken + rejected_count, 1)

    # Calculate average TIS from accepted messages
    tis_values = [
        m.get("metrics", {}).get("TIS", 0.0) if isinstance(m, dict)
        else getattr(m, "metrics", {}).get("TIS", 0.0)
        for m in accepted_messages
    ]
    avg_tis = sum(tis_values) / len(tis_values) if tis_values else 0.0

    # Try LLM-based extraction
    structured = _llm_extract_handoff(raw_deliverable, subgoal_name, phase_type, model, base_url)

    if structured:
        return HandoffPackage(
            subgoal_id=subgoal_id,
            subgoal_name=subgoal_name,
            phase_type=phase_type,
            what_was_done=structured.get("what_was_done", f"Completed {phase_type} phase for {subgoal_name}"),
            key_deliverable=raw_deliverable,
            what_matters_downstream=structured.get("what_matters_downstream", []),
            uncertainties=structured.get("uncertainties", []),
            suggested_focus=structured.get("suggested_focus", ""),
            turns_taken=turns_taken,
            acceptance_rate=acceptance_rate,
            avg_tis=avg_tis,
        )

    # Fallback: heuristic extraction
    return _heuristic_handoff(
        subgoal_id, subgoal_name, phase_type, raw_deliverable,
        turns_taken, acceptance_rate, avg_tis
    )


def _llm_extract_handoff(
    deliverable: str,
    subgoal_name: str,
    phase_type: str,
    model: str,
    base_url: str,
) -> Optional[Dict]:
    """Use LLM to extract structured handoff fields from raw deliverable."""
    system_prompt = (
        "Extract structured handoff information from the text below. "
        "Return valid JSON only, no markdown, no extra text."
    )
    user_prompt = (
        f"Subgoal: {subgoal_name} (phase: {phase_type})\n\n"
        f"Content:\n{deliverable[:2000]}\n\n"
        "Return JSON with this schema:\n"
        '{"what_was_done":"2-3 sentence summary",'
        '"what_matters_downstream":["point1","point2","point3"],'
        '"uncertainties":["issue1","issue2"],'
        '"suggested_focus":"what the next phase should prioritize"}'
    )

    try:
        resp = chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            base_url=base_url,
            timeout=20.0,
            max_tokens=200,
            temperature=0.1,
        )
        # Extract JSON from response
        raw = resp.text.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
    except Exception:
        pass
    return None


def _heuristic_handoff(
    subgoal_id: int,
    subgoal_name: str,
    phase_type: str,
    deliverable: str,
    turns_taken: int,
    acceptance_rate: float,
    avg_tis: float,
) -> HandoffPackage:
    """Fallback: extract handoff fields heuristically from raw text."""
    # Extract first sentence as summary
    sentences = re.split(r'[.!?]\s+', deliverable[:500])
    what_was_done = sentences[0] + "." if sentences else f"Completed {phase_type} for {subgoal_name}."

    # Extract bullet points as key findings
    bullets = re.findall(r'[-•*]\s+(.+?)(?:\n|$)', deliverable)
    key_points = bullets[:5] if bullets else [f"Completed {phase_type} analysis"]

    # Extract questions as uncertainties
    questions = re.findall(r'[?]\s*(.{10,60})\?', deliverable)
    uncertainties = questions[:3]

    return HandoffPackage(
        subgoal_id=subgoal_id,
        subgoal_name=subgoal_name,
        phase_type=phase_type,
        what_was_done=what_was_done,
        key_deliverable=deliverable,
        what_matters_downstream=key_points,
        uncertainties=uncertainties,
        suggested_focus=f"Build on the {phase_type} findings and address any gaps.",
        turns_taken=turns_taken,
        acceptance_rate=acceptance_rate,
        avg_tis=avg_tis,
    )
