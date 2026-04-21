"""
Tier-1: Emergent Team Formation (ETF) - FIXED FOR LOCAL MODELS

Simplified and robust team formation that works with smaller local models.
Uses predefined roles with clear behavioral differentiation.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict, field
import time
import json
import re

from .llm_client import chat_completion, LLMResponse
from .metrics import score_message
from .config import Agent, thresholds
from .subgoal_decomposer import Subgoal


@dataclass
class Role:
    """Represents a role in the team"""
    name: str
    description: str
    responsibilities: List[str]
    system_prompt_addition: str  # NEW: Specific prompt for this role


@dataclass
class TeamMember:
    """Represents an agent with an assigned role"""
    agent_id: str
    model: str
    role: Role
    confidence: float
    secondary_roles: List[Role] = None  # NEW: Support for multiple roles
    capabilities: List[str] = None  # NEW: For display in team table

    def __post_init__(self):
        if self.secondary_roles is None:
            self.secondary_roles = []
        if self.capabilities is None:
            # Auto-populate from role responsibilities
            self.capabilities = self.role.responsibilities[:3] if self.role else []


@dataclass
class TeamBlueprint:
    """Final team structure after formation"""
    members: List[TeamMember]
    roles: List[Role]
    subgoals: List[Subgoal]
    formation_turns: int
    negotiation_policy: Dict[str, Any] = field(default_factory=dict)
    consensus_strength: float = 0.0
    negotiation_quality: float = 0.0
    renegotiation_count: int = 0


# FIXED: Predefined roles that work well with local models and serve as fallback for DRE
# DYNAMIC: Roles are now generated per-task.
PREDEFINED_ROLES = {
    "Researcher": Role(
        name="Researcher",
        description="Focuses on information gathering, fact-checking, and exploring options.",
        responsibilities=["Gather information", "Verify facts", "Explore alternatives"],
        system_prompt_addition="You are a Researcher. Prioritize finding accurate information and listing options."
    ),
    "Analyst": Role(
        name="Analyst",
        description="Focuses on logical consistency, feasibility analysis, and data synthesis.",
        responsibilities=["Analyze data", "Check logical consistency", "Evaluate feasibility"],
        system_prompt_addition="You are an Analyst. Prioritize logic, feasibility, and data synthesis."
    ),
    "Writer": Role(
        name="Writer",
        description="Focuses on content structure, drafting, and clarity.",
        responsibilities=["Structure content", "Draft narrative", "Ensure clarity"],
        system_prompt_addition="You are a Writer. Prioritize clear structure and comprehensive drafting."
    ),
    "Reviewer": Role(
        name="Reviewer",
        description="Focuses on critique, quality assurance, and identifying gaps.",
        responsibilities=["Critique content", "Identify gaps", "Ensure quality"],
        system_prompt_addition="You are a Reviewer. Prioritize identifying missing information and logic gaps."
    )
}



class TeamFormation:
    """
    SIMPLIFIED team formation that works reliably with local models.
    Uses predefined roles and simple assignment logic.
    """
    
    def __init__(self, base_url: str, agents: List[Agent], max_formation_turns: int = 3, event_callback: Optional[callable] = None, memory: Optional[Any] = None, task_analysis: Optional[Dict[str, Any]] = None, is_renegotiation: bool = False, subgoals: Optional[List[Any]] = None):
        self.base_url = base_url
        self.agents = agents
        self.subgoals = subgoals or []
        self.max_formation_turns = max_formation_turns
        self._event_callback = event_callback
        self.memory = memory
        self.task_analysis = task_analysis or {}
        self.is_renegotiation = is_renegotiation

        # Simplified state
        self.formation_messages: List[Dict] = []
        self.agent_preferences: Dict[str, str] = {}  # agent_id -> preferred_role
        self.task_type: str = "general"
        self.available_roles: Dict[str, Role] = {} # Dynamic roles for the current task
        self.negotiation_policy: Dict[str, Any] = {}
        self.negotiation_scores: List[float] = []
        self.consensus_strength: float = 0.0
        self.negotiation_rounds: int = 0
        self.negotiation_quality_avg: float = 0.0
        self.secondary_assignments: Dict[str, List[str]] = {}
        self._negotiation_fail_streaks: Dict[str, int] = {}

    def _emit_event(self, event: Dict) -> None:
        if not self._event_callback:
            return
        try:
            self._event_callback(event)
        except Exception:
            pass
    def form_team(self, task_description: str) -> TeamBlueprint:
        """
        Execute team formation with adaptive negotiation policy and consensus gating.
        """
        if len(self.agents) == 1:
            return self._create_single_agent_blueprint(task_description)

        print("\n[Tier-1] Agents are negotiating roles through dialogue...")

        self.task_type = self._classify_task_type(task_description)
        print(f"[Tier-1] Task type identified: {self.task_type}")

        print("[Tier-1] Generating dynamic roles for this task...")
        self.available_roles = self._generate_custom_roles(task_description)
        self.secondary_assignments = {}
        print(f"[Tier-1] Generated Roles: {', '.join(self.available_roles.keys())}")

        self.negotiation_policy = self._build_negotiation_policy(task_description)
        self._emit_event({
            "type": "negotiation_policy",
            "timestamp": time.time(),
            "policy": self.negotiation_policy,
            "task_type": self.task_type,
            "is_renegotiation": self.is_renegotiation,
        })

        negotiation_success = self._conduct_team_negotiation(task_description)

        # FIX 5: Retry once if consensus is below the execution threshold
        min_exec = float(self.negotiation_policy.get("min_consensus_to_execute", 0.72))
        if not negotiation_success and not self.is_renegotiation and self.consensus_strength < min_exec:
            print(f"[Tier-1] LOW CONSENSUS ({self.consensus_strength:.2f} < {min_exec:.2f}), running retry round...")
            self._emit_event({
                "type": "negotiation_retry",
                "timestamp": time.time(),
                "consensus_strength": round(self.consensus_strength, 3),
                "threshold": min_exec,
            })
            self.max_formation_turns = min(self.max_formation_turns + 2, 8)
            negotiation_success = self._conduct_team_negotiation(task_description)
            if self.consensus_strength < min_exec:
                print(f"[Tier-1] ⚠ LOW CONSENSUS WARNING after retry ({self.consensus_strength:.2f}). Proceeding with caution.")
                self._emit_event({
                    "type": "low_consensus_warning",
                    "timestamp": time.time(),
                    "consensus_strength": round(self.consensus_strength, 3),
                    "threshold": min_exec,
                })

        if not negotiation_success:
            print("[Tier-1] Negotiation failed, applying fallback role coverage...")
            assigned_roles = set()
            for agent in self.agents:
                preferred_role = self.agent_preferences.get(agent.id)
                if preferred_role and preferred_role in self.available_roles and preferred_role not in assigned_roles:
                    assigned_roles.add(preferred_role)
                else:
                    for role_name in self.available_roles:
                        if role_name not in assigned_roles:
                            self.agent_preferences[agent.id] = role_name
                            assigned_roles.add(role_name)
                            break

        self.negotiation_quality_avg = (
            sum(self.negotiation_scores) / len(self.negotiation_scores)
            if self.negotiation_scores else 0.0
        )
        self._persist_negotiation_memory(reopened=self.is_renegotiation)

        blueprint = self._create_negotiated_team_blueprint(task_description)
        blueprint.negotiation_policy = dict(self.negotiation_policy)
        blueprint.consensus_strength = round(self.consensus_strength, 3)
        blueprint.negotiation_quality = round(self.negotiation_quality_avg, 3)
        blueprint.renegotiation_count = 1 if self.is_renegotiation else 0
        return blueprint


    def _build_negotiation_policy(self, task_description: str) -> Dict[str, Any]:
        """Build adaptive negotiation policy from task analysis and memory."""
        analysis = self.task_analysis or {}
        try:
            complexity_score = float(analysis.get("complexity_score", 0.45))
        except Exception:
            complexity_score = 0.45

        if not analysis:
            words = len(re.findall(r"\b\w+\b", task_description.lower()))
            markers = sum(1 for k in ["multi", "constraint", "trade", "risk", "phase"] if k in task_description.lower())
            complexity_score = max(0.2, min(1.0, (words / 260.0) + 0.08 * markers))

        min_rounds = 2 if complexity_score < 0.65 else 3
        max_rounds = min(self.max_formation_turns, max(min_rounds + 1, int(round(2 + complexity_score * 4))))
        consensus_threshold = max(0.68, min(0.9, 0.68 + 0.18 * complexity_score))
        quality_floor = max(0.35, min(0.8, 0.38 + 0.22 * complexity_score))
        stagnation_patience = 2 if complexity_score < 0.75 else 3

        memory_stats = {}
        if self.memory and hasattr(self.memory, "get_negotiation_patterns"):
            try:
                memory_stats = self.memory.get_negotiation_patterns(self.task_type) or {}
            except Exception:
                memory_stats = {}

        if memory_stats:
            avg_rounds = float(memory_stats.get("avg_rounds", max_rounds))
            avg_consensus = float(memory_stats.get("avg_consensus", consensus_threshold))
            reopen_rate = float(memory_stats.get("reopen_rate", 0.0))

            max_rounds = int(round(0.6 * max_rounds + 0.4 * avg_rounds))
            max_rounds = max(min_rounds + 1, min(self.max_formation_turns, max_rounds))

            if avg_consensus < 0.65:
                consensus_threshold = max(consensus_threshold, 0.74)
            if reopen_rate > 0.25:
                stagnation_patience = min(4, stagnation_patience + 1)

        return {
            "min_rounds": int(min_rounds),
            "max_rounds": int(max_rounds),
            "consensus_threshold": round(consensus_threshold, 3),
            "quality_floor": round(quality_floor, 3),
            "stagnation_patience": int(stagnation_patience),
            "min_consensus_to_execute": round(max(0.72, consensus_threshold - 0.04), 3),  # FIX 4: raised floor from 0.62 → 0.72
            "adaptive": True,
        }

    def _generate_custom_roles(self, task_description: str) -> Dict[str, Role]:
        """
        Use LLM to generate 4 specific roles tailored to the task, OR
        if subgoals are provided, directly map subgoals to roles to enable explicit DAG bidding.
        """
        if getattr(self, 'subgoals', None):
            roles = {}
            for sg in self.subgoals:
                role_name = sg.name
                roles[role_name] = Role(
                    name=role_name,
                    description=f"Responsible for executing the subgoal: {sg.description}",
                    responsibilities=[f"Complete {sg.phase_type} phase", sg.completion_criteria],
                    system_prompt_addition=f"You are responsible for the subgoal: {sg.name}."
                )
            return roles
        system_prompt = """You are a JSON generator. You output ONLY valid JSON.

⛔ NEGATIVE CONSTRAINTS (VIOLATION = FAILURE):
- NO <think> tags or thinking process
- NO markdown formatting (no ```json code blocks)
- NO introductory text (e.g. "Here is the JSON...")
- NO explanations
- NO trailing commas

✅ REQUIRED OUPUT FORMAT:
- A SINGLE line of raw JSON
- Strictly follows the schema below

SCHEMA:
{"roles": [{"name": "string", "description": "string", "responsibilities": ["string", "string"], "system_prompt_addition": "string"}, ...]}

EXAMPLE (Copy this structure exactly):
{"roles": [{"name": "Analyst", "description": "Analyzing data", "responsibilities": ["Data mining", "Report generation"], "system_prompt_addition": "Focus on accuracy."}, {"name": "Manager", "description": "Project oversight", "responsibilities": ["Coordination", "Risk mgmt"], "system_prompt_addition": "Prioritize deadlines."}, {"name": "Developer", "description": "Implementation", "responsibilities": ["Coding", "Testing"], "system_prompt_addition": "Write clean code."}, {"name": "Designer", "description": "UI/UX", "responsibilities": ["Wireframing", "Prototyping"], "system_prompt_addition": "Focus on usability."}]}

TASK: Generate 4 specific roles for the user prompt.
OUTPUT: Raw JSON only."""
        
        user_prompt = task_description

        try:
            # We use a smart model for this architectural decision
            architect_agent = self.agents[0] # Use the first agent's model as the architect
            
            response = chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=architect_agent.model,
                base_url=self.base_url,
                timeout=45.0,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Extract JSON
            text = response.text.strip()
            
            # Clean up potential markdown formatting
            text = re.sub(r"```[a-zA-Z]*", "", text, flags=re.IGNORECASE).strip().rstrip("`").strip()
            
            # Find the first '{' and last '}'
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx+1]
                # Fix common JSON errors
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                
                # Sanitize control characters that break JSON
                # This regex finds newlines inside quotes and escapes them
                # But a simpler approach for now is to trust the LLM mostly and just handle the specific error
                # We can use strict=False which helps sometimes
                # NUCLEAR JSON REPAIR - 5 strategies
                data = None
                errors = []
                
                # Strategy 1: Direct parse
                try:
                    data = json.loads(json_str, strict=False)
                except json.JSONDecodeError as e1:
                    errors.append(f"S1: {str(e1)[:80]}")
                    
                    # Strategy 2: Remove trailing commas
                    try:
                        json_clean = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        data = json.loads(json_clean, strict=False)
                    except json.JSONDecodeError as e2:
                        errors.append(f"S2: {str(e2)[:80]}")
                        
                        # Strategy 3: Add missing commas between array elements
                        try:
                            json_clean = re.sub(r'}\s*\n\s*{', '},\n{', json_str)
                            data = json.loads(json_clean, strict=False)
                        except json.JSONDecodeError as e3:
                            errors.append(f"S3: {str(e3)[:80]}")
                            
                            # Strategy 4: Fix unescaped characters
                            try:
                                json_clean = re.sub(
                                    r':\s*"([^"]*?)"',
                                    lambda m: ': "' + m.group(1).replace('\n', '\\n').replace('\r', '').replace('\t', '\\t').replace('\\', '\\\\') + '"',
                                    json_str
                                )
                                data = json.loads(json_clean, strict=False)
                            except json.JSONDecodeError as e4:
                                errors.append(f"S4: {str(e4)[:80]}")
                                
                                # Strategy 5: Regex extraction (nuclear option)
                                try:
                                    import ast
                                    # Try to extract role data manually
                                    role_pattern = r'"name"\s*:\s*"([^"]+)".*?"description"\s*:\s*"([^"]+)".*?"responsibilities"\s*:\s*\[([^\]]+)\].*?"system_prompt_addition"\s*:\s*"([^"]+)"'
                                    matches = re.findall(role_pattern, json_str, re.DOTALL)
                                    if matches:
                                        roles_list = []
                                        # Limit to maximum 4 roles
                                        for name, desc, resp_str, prompt in matches[:4]:
                                            # Parse responsibilities array
                                            resp_items = [r.strip().strip('"').strip(',') for r in resp_str.split('"') if r.strip() and r.strip() not in [',', '']]
                                            roles_list.append({
                                                "name": name,
                                                "description": desc,
                                                "responsibilities": [r for r in resp_items if r],
                                                "system_prompt_addition": prompt
                                            })
                                        data = {"roles": roles_list}
                                        print(f"[Tier-1] ⚠ Used regex extraction (Strategy 5) - Extracted {len(roles_list)} roles")
                                    else:
                                        raise ValueError("No role data found")
                                except Exception as e5:
                                    errors.append(f"S5: {str(e5)[:80]}")
                                    print(f"[Tier-1] ✗ ALL 5 STRATEGIES FAILED:")
                                    for i, err in enumerate(errors, 1):
                                        print(f"  {i}. {err}")
                                    print(f"[Tier-1] Raw JSON (first 1000 chars):\n{json_str[:1000]}\n...")
                                    raise ValueError(f"JSON parsing failed: {e1}") from e1
            else:
                # Try simple regex as fallback
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                else:
                    raise ValueError("No JSON found in response")
            
            if data:
                role_objects = {}
                for i, r_data in enumerate(data.get("roles", [])):
                    if not isinstance(r_data, dict):
                        continue

                    # Be permissive: local models often return partial role objects.
                    role_name = str(r_data.get("name") or f"Role{i + 1}").strip()
                    if not role_name:
                        role_name = f"Role{i + 1}"

                    description = str(
                        r_data.get("description")
                        or f"Handles {role_name.lower()} responsibilities for this task."
                    ).strip()

                    responsibilities_raw = (
                        r_data.get("responsibilities")
                        or r_data.get("responsibility")
                        or r_data.get("capabilities")
                        or r_data.get("tasks")
                    )
                    responsibilities: List[str] = []

                    if isinstance(responsibilities_raw, list):
                        responsibilities = [
                            str(item).strip() for item in responsibilities_raw if str(item).strip()
                        ]
                    elif isinstance(responsibilities_raw, str):
                        responsibilities = [
                            part.strip()
                            for part in re.split(r"[;\n,]+", responsibilities_raw)
                            if part.strip()
                        ]

                    if not responsibilities:
                        responsibilities = [
                            f"Define {role_name} scope and plan",
                            f"Execute {role_name} tasks",
                            f"Report {role_name} risks and decisions",
                        ]

                    system_prompt_addition = str(
                        r_data.get("system_prompt_addition")
                        or r_data.get("prompt")
                        or f"You are {role_name}. Focus on: {', '.join(responsibilities[:3])}."
                    ).strip()

                    role_objects[role_name] = Role(
                        name=role_name,
                        description=description,
                        responsibilities=responsibilities,
                        system_prompt_addition=system_prompt_addition,
                    )

                # Verify we have at least 3 roles
                if len(role_objects) >= 3:
                    print(f"[Tier-1] ✓ Successfully generated {len(role_objects)} dynamic roles!")
                    self._emit_event({
                        "type": "roles_generated",
                        "timestamp": time.time(),
                        "roles": [asdict(role) for role in role_objects.values()]
                    })
                    return role_objects
                else:
                    raise ValueError(f"Only {len(role_objects)} roles generated, need at least 3")

            raise ValueError("No valid role data in response")

        except Exception as e:
            # FIX 3: Graceful fallback to PREDEFINED_ROLES instead of crashing
            print(f"[Tier-1] ✗ Dynamic role generation FAILED: {e}")
            print(f"[Tier-1] ⚠ Falling back to PREDEFINED_ROLES")
            self._emit_event({
                "type": "role_generation_fallback",
                "timestamp": time.time(),
                "reason": str(e),
                "roles": [{"name": name, "description": r.description} for name, r in PREDEFINED_ROLES.items()]
            })
            return dict(PREDEFINED_ROLES)
    def _conduct_team_negotiation(self, task_description: str) -> bool:
        """
        Conduct multi-turn negotiation with adaptive stopping based on
        consensus strength and turn quality.
        """
        print("[Tier-1] Starting role negotiation dialogue...")

        roles_text = "\n".join([f"- {r.name}: {r.description}" for r in self.available_roles.values()])
        negotiation_context = self._build_slim_negotiation_context(task_description, roles_text)
        self._negotiation_fail_streaks = {agent.id: 0 for agent in self.agents}

        min_rounds = int(self.negotiation_policy.get('min_rounds', 2))
        max_rounds = int(self.negotiation_policy.get('max_rounds', self.max_formation_turns))
        consensus_threshold = float(self.negotiation_policy.get('consensus_threshold', 0.72))
        quality_floor = float(self.negotiation_policy.get('quality_floor', 0.45))
        stagnation_patience = int(self.negotiation_policy.get('stagnation_patience', 2))

        try:
            print("\n[Tier-1] Turn 1: Initial role proposals")
            proposals = {}
            for agent in self.agents:
                proposal = self._get_role_proposal(agent, negotiation_context, [])
                if proposal.get("ok", True):
                    self._negotiation_fail_streaks[agent.id] = 0
                else:
                    self._negotiation_fail_streaks[agent.id] = self._negotiation_fail_streaks.get(agent.id, 0) + 1
                proposals[agent.id] = proposal

                message = {
                    "turn": 1,
                    "agent": agent.id,
                    "content": f"I propose to take the {proposal['role']} role. {proposal['reasoning']}",
                    "proposed_role": proposal['role'],
                    "negotiation_score": 0.0,
                    "phase": "primary"
                }
                self.formation_messages.append(message)
                self._emit_event({"type": "negotiation_turn", "timestamp": time.time(), **message})

            current = {aid: {"final_role": p["role"], "reasoning": p["reasoning"]} for aid, p in proposals.items()}
            final_assignments, uncovered_roles = self._build_consensus(current, task_description)
            self.consensus_strength = self._compute_consensus_strength(final_assignments, current)
            self.negotiation_rounds = 1
            prev_strength = self.consensus_strength
            stagnation = 0

            for round_idx in range(2, max_rounds + 1):
                print(f"\n[Tier-1] Turn {round_idx}: negotiation updates")
                responses = {}
                for agent in self.agents:
                    own = {
                        "role": current.get(agent.id, {}).get("final_role", list(self.available_roles.keys())[0]),
                        "reasoning": current.get(agent.id, {}).get("reasoning", ""),
                    }
                    merged_context = {**current}
                    merged_context.update(responses)
                    other = {
                        aid: {"role": item.get("final_role", ""), "reasoning": item.get("reasoning", "")}
                        for aid, item in merged_context.items() if aid != agent.id
                    }
                    response = self._get_negotiation_response(agent, negotiation_context, own, other)
                    if response.get("ok", True):
                        self._negotiation_fail_streaks[agent.id] = 0
                    else:
                        self._negotiation_fail_streaks[agent.id] = self._negotiation_fail_streaks.get(agent.id, 0) + 1

                    if self._negotiation_fail_streaks.get(agent.id, 0) >= 2:
                        reassigned_role = self._choose_failover_role(agent.id, own.get("role", ""), current, responses)
                        response = {
                            "final_role": reassigned_role,
                            "reasoning": f"Auto-reassigned to {reassigned_role} after 2 consecutive negotiation failures.",
                            "ok": True,
                        }
                        self._negotiation_fail_streaks[agent.id] = 0
                        self._emit_event({
                            "type": "fail_fast_role_reassign",
                            "timestamp": time.time(),
                            "agent_id": agent.id,
                            "new_role": reassigned_role,
                            "reason": "2_consecutive_negotiation_failures",
                            "turn": round_idx,
                        })

                    responses[agent.id] = response

                    score = self._score_negotiation_turn(agent.id, response.get("reasoning", ""), response.get("final_role", own["role"]), current)
                    self.negotiation_scores.append(score)

                    message = {
                        "turn": round_idx,
                        "agent": agent.id,
                        "content": response.get("reasoning", ""),
                        "final_role": response.get("final_role", own["role"]),
                        "negotiation_score": round(score, 3),
                        "phase": "primary"
                    }
                    self.formation_messages.append(message)
                    self._emit_event({"type": "negotiation_turn", "timestamp": time.time(), **message})
                    self._emit_event({
                        "type": "negotiation_quality",
                        "timestamp": time.time(),
                        "turn": round_idx,
                        "agent_id": agent.id,
                        "score": round(score, 3),
                    })

                current = responses
                final_assignments, uncovered_roles = self._build_consensus(current, task_description)
                self.consensus_strength = self._compute_consensus_strength(final_assignments, current)
                self.negotiation_rounds = round_idx

                recent_scores = self.negotiation_scores[-max(1, len(self.agents)):]
                recent_quality = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0

                self._emit_event({
                    "type": "consensus_meter",
                    "timestamp": time.time(),
                    "turn": round_idx,
                    "consensus_strength": round(self.consensus_strength, 3),
                    "recent_turn_quality": round(recent_quality, 3),
                })

                if round_idx >= min_rounds and self.consensus_strength >= consensus_threshold and recent_quality >= quality_floor:
                    print(f"[Tier-1] Converged at turn {round_idx} (consensus={self.consensus_strength:.2f}).")
                    break

                if self.consensus_strength <= prev_strength + 0.01:
                    stagnation += 1
                else:
                    stagnation = 0
                prev_strength = self.consensus_strength

                if round_idx >= min_rounds and stagnation >= stagnation_patience:
                    print(f"[Tier-1] Stopped due to stagnation at turn {round_idx}.")
                    self._emit_event({
                        "type": "negotiation_stop",
                        "timestamp": time.time(),
                        "reason": "stagnation",
                        "turn": round_idx,
                        "consensus_strength": round(self.consensus_strength, 3),
                    })
                    break

            # ── Supervisor Override ────────────────────────────────────────
            # If we exit the loop without strong consensus, inject a final
            # "Supervisor" prompt that forces each agent to commit to a role
            # in JSON. This breaks infinite polite-agreement loops.
            if self.consensus_strength < consensus_threshold:
                print(f"[Tier-1] ⚠ Supervisor Override: consensus={self.consensus_strength:.2f} < "
                      f"threshold={consensus_threshold:.2f}. Forcing final role commit...")
                role_names = list(self.available_roles.keys())
                supervisor_prompt = (
                    f"SUPERVISOR: The negotiation must conclude now. "
                    f"Based on all previous discussion, commit to your FINAL role assignment. "
                    f"Reply ONLY with a JSON object: "
                    f'{{ "final_role": "<one of: {", ".join(role_names)}>", '
                    f'"reasoning": "<one sentence why>" }}'
                )
                forced_responses = {}
                for agent in self.agents:
                    try:
                        supervisor_msgs = [
                            {"role": "system", "content": "You are a negotiation participant. Follow supervisor instructions exactly."},
                            {"role": "user", "content": supervisor_prompt},
                        ]
                        resp = chat_completion(
                            messages=supervisor_msgs,
                            base_url=self.base_url,
                            model=agent.model,
                            max_tokens=120,
                            temperature=0.1,
                        )
                        text = resp.text.strip() if resp else ""
                        import re as _re
                        m = _re.search(r'\{.*\}', text, _re.DOTALL)
                        parsed = json.loads(m.group(0)) if m else {}
                        forced_responses[agent.id] = {
                            "final_role": parsed.get("final_role", role_names[0]),
                            "reasoning": parsed.get("reasoning", "supervisor override"),
                        }
                    except Exception as sv_err:
                        print(f"[Tier-1] Supervisor override failed for {agent.id}: {sv_err}")
                        forced_responses[agent.id] = current.get(agent.id, {"final_role": role_names[0], "reasoning": ""})

                current = forced_responses
                final_assignments, uncovered_roles = self._build_consensus(current, task_description)
                self.consensus_strength = self._compute_consensus_strength(final_assignments, current)
                print(f"[Tier-1] ✓ Supervisor Override complete. consensus={self.consensus_strength:.2f}")
            # ──────────────────────────────────────────────────────────────

            if uncovered_roles:
                secondary_map = self._negotiate_secondary_roles(
                    uncovered_roles=uncovered_roles,
                    final_assignments=final_assignments,
                    primary_responses=current,
                    task_description=task_description,
                )
                for aid, roles in secondary_map.items():
                    if aid in final_assignments:
                        final_assignments[aid]["secondary"] = list(dict.fromkeys(roles))

            self.secondary_assignments = {aid: info.get("secondary", []) for aid, info in final_assignments.items()}

            for agent_id, role_info in final_assignments.items():
                self.agent_preferences[agent_id] = role_info['role']
                secondaries = role_info.get("secondary", []) or []
                if secondaries:
                    self.agent_preferences[agent_id + "_secondary"] = secondaries
                elif (agent_id + "_secondary") in self.agent_preferences:
                    del self.agent_preferences[agent_id + "_secondary"]

            self._print_final_team_roles(final_assignments)

            return len(final_assignments) == len(self.agents) and self.consensus_strength >= float(self.negotiation_policy.get('min_consensus_to_execute', 0.62))

        except Exception as e:
            print(f"[Tier-1] Negotiation failed with error: {e}")
            return False

    def _get_role_proposal(self, agent: Agent, context: str, previous_messages: List) -> Dict[str, str]:
        """Get initial role proposal from an agent"""
        role_names = "/".join(self.available_roles.keys())
        system_prompt = f"""You are participating in team formation for a collaborative task.

{context}

Propose which role you want to take and explain why you're the best fit.
Be specific about your strengths and how they align with the role requirements.

Respond in this format:
Role: [{role_names}]
Reasoning: [1 short sentence explaining fit]

IMPORTANT: Do NOT select a role that is already well-covered if another role is open. Ensure ALL roles are covered."""

        try:
            response = chat_completion(
                system_prompt=system_prompt,
                user_prompt="What role do you propose to take for this task?",
                model=agent.model,
                base_url=self.base_url,
                timeout=30.0,
                max_tokens=160,
                temperature=0.5
            )
            
            # Parse response
            text = response.text.strip()
            role_match = re.search(r'Role:\s*(\w+)', text, re.IGNORECASE)
            reasoning_match = re.search(r'Reasoning:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
            
            role = role_match.group(1) if role_match else "Researcher"
            reasoning = reasoning_match.group(1).strip() if reasoning_match else text
            
            # Validate role
            valid_roles = list(self.available_roles.keys())
            if role not in valid_roles:
                # Find closest match
                role_lower = role.lower()
                for valid_role in valid_roles:
                    if valid_role.lower() in role_lower or role_lower in valid_role.lower():
                        role = valid_role
                        break
                else:
                    role = valid_roles[0]  # Default fallback

            
            return {"role": role, "reasoning": reasoning[:260], "ok": True}
            
        except Exception as e:
            print(f"[Tier-1] Role proposal failed for {agent.id}: {e}")
            default_role = list(self.available_roles.keys())[0] if self.available_roles else "Researcher"
            return {"role": default_role, "reasoning": "Default assignment due to error", "ok": False}
    
    def _get_negotiation_response(self, agent: Agent, context: str, own_proposal: Dict, other_proposals: Dict) -> Dict[str, str]:
        """Get agent's response to other proposals and final role decision"""
        other_proposals_text = "\n".join([
            f"- {aid}: {prop.get('role', 'Unknown')} ({self._trim_text(prop.get('reasoning', ''), 80)})"
            for aid, prop in list(other_proposals.items())[:4]
        ])
        
        role_names = "/".join(self.available_roles.keys())
        system_prompt = f"""You are in team formation negotiations.

{context}

Your initial proposal: {own_proposal['role']} - {own_proposal['reasoning']}

Other agents' proposals:
{other_proposals_text}

Consider the other proposals and decide on your final role choice.
You can:
1. Stick with your original choice if it makes sense
2. Switch to a different role if there's conflict or better fit
3. Suggest role swaps if beneficial

Respond in this format:
Final Role: [{role_names}]
Reasoning: [one short sentence considering team balance]

CRITICAL: If multiple agents want the same role, consider switching to an uncovered role to ensure the team succeeds. Diversity is Key."""

        try:
            response = chat_completion(
                system_prompt=system_prompt,
                user_prompt="What is your final role decision after seeing other proposals?",
                model=agent.model,
                base_url=self.base_url,
                timeout=30.0,
                max_tokens=180,
                temperature=0.4
            )
            
            # Parse response
            text = response.text.strip()
            role_match = re.search(r'Final Role:\s*(\w+)', text, re.IGNORECASE)
            reasoning_match = re.search(r'Reasoning:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
            
            role = role_match.group(1) if role_match else own_proposal['role']
            reasoning = reasoning_match.group(1).strip() if reasoning_match else text
            
            # Validate role
            valid_roles = list(self.available_roles.keys())
            if role not in valid_roles:
                role = own_proposal['role']  # Fallback to original
            
            return {"final_role": role, "reasoning": reasoning[:280], "ok": True}
            
        except Exception as e:
            print(f"[Tier-1] Negotiation response failed for {agent.id}: {e}")
            return {"final_role": own_proposal['role'], "reasoning": "Keeping original role due to error", "ok": False}

    def _build_slim_negotiation_context(self, task_description: str, roles_text: str) -> str:
        """Build a compact context for negotiation rounds to reduce token pressure."""
        task_short = self._trim_text(task_description, 320)
        role_lines = [line for line in roles_text.splitlines() if line.strip()]
        compact_roles = "\n".join(self._trim_text(line, 96) for line in role_lines[:8])
        return (
            f"Task: {task_short}\n\n"
            f"Available roles:\n{compact_roles}\n\n"
            "Goal: one primary role per agent, maximize role coverage, avoid duplicates."
        )

    def _trim_text(self, text: str, max_len: int) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(cleaned) <= max_len:
            return cleaned
        return cleaned[: max_len - 3].rstrip() + "..."

    def _choose_failover_role(
        self,
        agent_id: str,
        current_role: str,
        current_assignments: Dict[str, Dict[str, str]],
        in_round_responses: Dict[str, Dict[str, str]],
    ) -> str:
        """Pick a less-contended role immediately when an agent is unstable."""
        role_counts: Dict[str, int] = {r: 0 for r in self.available_roles.keys()}
        merged = dict(current_assignments or {})
        merged.update(in_round_responses or {})
        for aid, info in merged.items():
            if aid == agent_id:
                continue
            r = info.get("final_role") or info.get("role")
            if r in role_counts:
                role_counts[r] += 1

        sorted_roles = sorted(role_counts.items(), key=lambda kv: (kv[1], kv[0]))
        for role_name, _ in sorted_roles:
            if role_name != current_role:
                return role_name
        return current_role or (list(self.available_roles.keys())[0] if self.available_roles else "Researcher")
    
    def _build_consensus(self, responses: Dict, task_description: str) -> Tuple[Dict[str, Dict], List[str]]:
        """
        Build final consensus on role assignments, prioritizing role diversity.
        Refined to support DYNAMIC roles and secondary role assignments.
        """
        # 1. Count role claims
        role_claims = {} # role_name -> [agent_ids]
        agent_claims = {} # agent_id -> role_name
        
        for agent_id, response in responses.items():
            role = response['final_role']
            agent_claims[agent_id] = role
            if role not in role_claims:
                role_claims[role] = []
            role_claims[role].append(agent_id)
            
        final_assignments = {}
        assigned_agents = set()
        covered_roles = set()
        
        # 2. Priority List: Use dynamic roles available
        # Sort roles by scarcity (if we could detect importance, but random/alpha is fine for now)
        # Ideally, we follow the order they were generated in (often priority order)
        priority_roles = list(self.available_roles.keys())
        
        # 3. Assign Uncontested Roles
        for role in priority_roles:
            if role in role_claims and len(role_claims[role]) == 1:
                agent_id = role_claims[role][0]
                if agent_id not in assigned_agents:
                    final_assignments[agent_id] = {
                        "role": role,
                        "confidence": 0.95,
                        "reasoning": responses[agent_id]['reasoning'],
                        "secondary": []
                    }
                    assigned_agents.add(agent_id)
                    covered_roles.add(role)
        
        # 4. Resolve Contested Roles (Multiple agents want same role)
        for role in priority_roles:
            if role in role_claims and len(role_claims[role]) > 1:
                # Find best candidate among claimants who aren't assigned yet
                candidates = [aid for aid in role_claims[role] if aid not in assigned_agents]

                if candidates:
                    # FIX 1: Score each claimant's reasoning quality; highest scorer wins
                    best_candidate = candidates[0]
                    best_score = -1.0
                    for cand in candidates:
                        reasoning_text = responses.get(cand, {}).get(
                            'reasoning', responses.get(cand, {}).get('final_role', '')
                        )
                        other_context = {
                            aid: {'final_role': responses.get(aid, {}).get('final_role', '')}
                            for aid in responses if aid != cand
                        }
                        score = self._score_negotiation_turn(cand, reasoning_text, role, other_context)
                        if score > best_score:
                            best_score = score
                            best_candidate = cand
                    winner = best_candidate
                    print(f"[Tier-1] Contested role '{role}' → {winner} wins (score: {best_score:.2f})")

                    final_assignments[winner] = {
                        "role": role,
                        "confidence": 0.9,
                        "reasoning": responses[winner]['reasoning'],
                        "secondary": []
                    }
                    assigned_agents.add(winner)
                    covered_roles.add(role)

                    # The losers are floated to the cleanup phase below
                    losers = [c for c in candidates if c != winner]
                    for loser in losers:
                        pass  # handled in step 5

        # 5. Assign Remaining Agents to Uncovered Roles
        unassigned_agents = [a.id for a in self.agents if a.id not in assigned_agents]
        uncovered_roles = [r for r in priority_roles if r not in covered_roles]
        
        for i, agent_id in enumerate(unassigned_agents):
            # Pick an uncovered role if available
            if i < len(uncovered_roles):
                role = uncovered_roles[i]
                reasoning = f"Reassigned to {role} to ensure full team coverage."
            else:
                # If all roles covered, double up on the first/most important role
                role = priority_roles[0]
                reasoning = f"Assigned to support {role}."
            
            final_assignments[agent_id] = {
                "role": role,
                "confidence": 0.7,
                "reasoning": reasoning,
                "secondary": []
            }
            assigned_agents.add(agent_id)
            covered_roles.add(role)

        # 6. Return primary assignments and uncovered roles.
        truly_uncovered = [r for r in priority_roles if r not in covered_roles]
        return final_assignments, truly_uncovered
    

    def _negotiate_secondary_roles(self, uncovered_roles: List[str], final_assignments: Dict[str, Dict[str, Any]], primary_responses: Dict[str, Dict[str, str]], task_description: str) -> Dict[str, List[str]]:
        """Run a short second negotiation for leftover roles."""
        print(f"[Tier-1] Secondary negotiation round for uncovered roles: {', '.join(uncovered_roles)}")

        secondary_map: Dict[str, List[str]] = {aid: [] for aid in final_assignments.keys()}
        if not uncovered_roles or not final_assignments:
            return secondary_map

        role_votes: Dict[str, List[Tuple[str, str]]] = {r: [] for r in uncovered_roles}
        role_names = "/".join(uncovered_roles)

        for agent in self.agents:
            if agent.id not in final_assignments:
                continue
            primary_role = final_assignments[agent.id].get("role", "")
            rationale_hint = primary_responses.get(agent.id, {}).get("reasoning", "") if primary_responses else ""

            system_prompt = f"""You are in a SHORT secondary-role negotiation round.
Task: {task_description}
Your primary role: {primary_role}
Uncovered roles: {', '.join(uncovered_roles)}

Pick at most ONE secondary role that complements your primary role.
If none fit, return NONE.

Respond exactly in this format:
Secondary Role: [{role_names}/NONE]
Reasoning: [one short sentence]
"""

            try:
                resp = chat_completion(
                    system_prompt=system_prompt,
                    user_prompt=f"Your prior reasoning: {rationale_hint[:220]}",
                    model=agent.model,
                    base_url=self.base_url,
                    timeout=20.0,
                    max_tokens=120,
                    temperature=0.4,
                )
                txt = (resp.text or "").strip()
                m_role = re.search(r"Secondary\s*Role:\s*([A-Za-z0-9_\- ]+)", txt, re.IGNORECASE)
                m_reason = re.search(r"Reasoning:\s*(.+)", txt, re.IGNORECASE | re.DOTALL)
                picked = m_role.group(1).strip() if m_role else "NONE"
                reason = (m_reason.group(1).strip() if m_reason else txt)[:220]

                normalized = next((r for r in uncovered_roles if r.lower() == picked.lower()), None)
                if normalized:
                    role_votes[normalized].append((agent.id, reason))
                    
                message = {
                    "turn": 1,
                    "agent": agent.id,
                    "content": reason,
                    "proposed_role": normalized if normalized else "NONE",
                    "final_role": normalized if normalized else "NONE",
                    "negotiation_score": 1.0,
                    "phase": "secondary"
                }
                self.formation_messages.append(message)
                self._emit_event({"type": "negotiation_turn", "timestamp": time.time(), **message})
            except Exception:
                continue

        assigned_once: Set[str] = set()
        for role in uncovered_roles:
            votes = role_votes.get(role, [])
            if votes:
                chosen_agent = votes[0][0]
                if chosen_agent not in assigned_once:
                    secondary_map[chosen_agent].append(role)
                    assigned_once.add(chosen_agent)
                    print(f"[Tier-1] Secondary role negotiated: {chosen_agent} -> {role}")
                    continue

            target_agent = min(
                secondary_map.keys(),
                key=lambda aid: len(secondary_map[aid]) + (1 if final_assignments.get(aid, {}).get("role") == role else 0)
            )
            secondary_map[target_agent].append(role)
            print(f"[Tier-1] Secondary role fallback: {target_agent} -> {role}")

        return secondary_map
    def _print_final_team_roles(self, final_assignments: Dict[str, Dict[str, Any]]) -> None:
        """Print final primary/secondary role map once after convergence."""
        print("[Tier-1] Final Team Role Map:")
        for agent in self.agents:
            info = final_assignments.get(agent.id, {})
            primary = info.get("role", "Unassigned")
            secondary = ", ".join(info.get("secondary", []) or [])
            if secondary:
                print(f"  - {agent.id}: Primary={primary} | Secondary={secondary}")
            else:
                print(f"  - {agent.id}: Primary={primary} | Secondary=-")

    def _score_negotiation_turn(self, agent_id: str, reasoning: str, selected_role: str, current_assignments: Dict[str, Dict[str, str]]) -> float:
        """Score negotiation turn quality to suppress low-value negotiation chatter."""
        text = (reasoning or "").strip()
        words = re.findall(r"\b\w+\b", text.lower())
        info_gain = min(1.0, len(words) / 110.0)

        prior_texts = [m.get("content", "") for m in self.formation_messages[-10:]]
        sims = [self._lex_jaccard(text, t) for t in prior_texts if t]
        max_sim = max(sims) if sims else 0.0

        novelty = 1.0 - max_sim
        redundancy = max_sim

        same_role_count = 0
        for aid, info in current_assignments.items():
            if aid == agent_id:
                continue
            if info.get("final_role") == selected_role:
                same_role_count += 1
        conflict_reduction = 1.0 / (1.0 + same_role_count)

        score = 0.35 * info_gain + 0.35 * novelty + 0.30 * conflict_reduction - 0.25 * redundancy
        return max(0.0, min(1.0, score))

    def _compute_consensus_strength(self, final_assignments: Dict[str, Dict[str, Any]], responses: Dict[str, Dict[str, str]]) -> float:
        """Compute consensus strength meter for execution gating."""
        if not final_assignments:
            return 0.0

        roles = [v.get("role", "") for v in final_assignments.values() if v.get("role")]
        unique_roles = set(roles)
        target_coverage = min(len(self.available_roles), len(self.agents))
        coverage = len(unique_roles) / max(1, target_coverage)

        conflict_ratio = max(0.0, (len(roles) - len(unique_roles)) / max(1, len(roles)))
        avg_confidence = sum(v.get("confidence", 0.7) for v in final_assignments.values()) / max(1, len(final_assignments))

        recent = self.negotiation_scores[-max(1, len(self.agents)):]
        turn_quality = sum(recent) / len(recent) if recent else 0.0

        strength = 0.40 * coverage + 0.25 * (1.0 - conflict_ratio) + 0.20 * avg_confidence + 0.15 * turn_quality
        return max(0.0, min(1.0, strength))

    def _lex_jaccard(self, a: str, b: str) -> float:
        wa = set(re.findall(r"\b[a-z0-9]+\b", (a or "").lower()))
        wb = set(re.findall(r"\b[a-z0-9]+\b", (b or "").lower()))
        if not wa and not wb:
            return 1.0
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / max(1, len(wa | wb))

    def _persist_negotiation_memory(self, reopened: bool = False) -> None:
        if not self.memory or not hasattr(self.memory, "update_negotiation_patterns"):
            return
        try:
            self.memory.update_negotiation_patterns(
                self.task_type,
                {
                    "rounds": float(self.negotiation_rounds),
                    "consensus_strength": float(self.consensus_strength),
                    "turn_quality": float(self.negotiation_quality_avg if self.negotiation_scores else 0.0),
                    "reopened": bool(reopened),
                },
            )
        except Exception:
            pass

    def _create_negotiated_team_blueprint(self, task_description: str) -> TeamBlueprint:
        """Create team blueprint based on negotiation results"""
        members = []
        roles_used = []
        
        for agent in self.agents:
            role_name = self.agent_preferences.get(agent.id)
            if not role_name or role_name not in self.available_roles:
                # Fallback if something went wrong
                role_name = list(self.available_roles.keys())[0]
            
            role = self.available_roles[role_name]

            
            # Find confidence from formation messages
            confidence = 0.9
            for msg in self.formation_messages:
                if msg.get("agent") == agent.id and "final_role" in msg:
                    confidence = 0.9 if msg["final_role"] == role_name else 0.7
                    break
            
            secondary_names = self.agent_preferences.get(agent.id + "_secondary", []) or self.secondary_assignments.get(agent.id, [])
            secondary_roles = [self.available_roles[r] for r in secondary_names if r in self.available_roles]

            members.append(TeamMember(
                agent_id=agent.id,
                model=agent.model,
                role=role,
                confidence=confidence,
                secondary_roles=secondary_roles
            ))
            roles_used.append(role)
        
        # Create subgoals based on negotiated team structure
        subgoals = self._create_task_appropriate_subgoals(task_description)
        
        return TeamBlueprint(
            members=members,
            roles=roles_used,
            subgoals=subgoals,
            formation_turns=len(self.formation_messages),
            negotiation_policy=dict(self.negotiation_policy),
            consensus_strength=float(self.consensus_strength),
            negotiation_quality=float(self.negotiation_quality_avg),
            renegotiation_count=1 if self.is_renegotiation else 0,
        )
    
    def _create_task_appropriate_subgoals(self, task_description: str) -> List[Subgoal]:
        """Create subgoals based on task type.
        
        When 2+ distinct Tier-1 roles are available, generates parallel
        research branches (one per role, all with deps=[]) followed by
        a merge analysis, draft, and review phase.  This enables true
        DAG parallel execution.
        """
        task_type = self._classify_task_type(task_description)
        
        # ── Role-aware parallel DAG generation ──────────────────────
        roles = list(getattr(self, 'available_roles', {}).values())
        if len(roles) >= 2:
            subgoals: List[Subgoal] = []
            role_ids: List[int] = []
            
            # One parallel research branch per Tier-1 role
            for idx, role in enumerate(roles, start=1):
                subgoals.append(Subgoal(
                    id=idx,
                    name=f"{role.name} Research",
                    description=f"Research from the {role.name} perspective: {role.description}",
                    completion_criteria=f"{role.name} perspective fully explored",
                    estimated_turns=5,
                    phase_type="research",
                    dependencies=[],   # No deps → runs in parallel
                ))
                role_ids.append(idx)
            
            next_id = len(roles) + 1
            
            # Merge analysis — depends on ALL parallel research branches
            subgoals.append(Subgoal(
                id=next_id,
                name="Combined Analysis",
                description="Synthesize findings from all parallel research branches into a unified analysis",
                completion_criteria="All perspectives integrated into framework",
                estimated_turns=5,
                phase_type="analysis",
                dependencies=role_ids,  # Waits for every research branch
            ))
            
            # Draft — depends on analysis
            subgoals.append(Subgoal(
                id=next_id + 1,
                name="Draft Document",
                description="Write comprehensive deliverable incorporating all perspectives",
                completion_criteria="Complete document drafted",
                estimated_turns=8,
                phase_type="draft",
                dependencies=[next_id],
            ))
            
            # Review — depends on draft
            subgoals.append(Subgoal(
                id=next_id + 2,
                name="Final Review",
                description="Review deliverable for completeness, accuracy, and quality",
                completion_criteria="Deliverable reviewed and finalized",
                estimated_turns=4,
                phase_type="review",
                dependencies=[next_id + 1],
            ))
            
            print(f"[Tier-2] DAG: {len(roles)} parallel research branches + merge/draft/review")
            return subgoals
        
        # ── Fallback: linear template (single role or no roles) ─────
        if task_type == "policy":
            return [
                Subgoal(1, "Requirements Research", "Gather policy requirements and regulations", "Requirements documented", 6, "research", []),
                Subgoal(2, "Framework Analysis", "Analyze existing policies and create framework", "Framework established", 5, "analysis", [1]),
                Subgoal(3, "Policy Drafting", "Create comprehensive policy document", "Policy document complete", 8, "draft", [1]),
                Subgoal(4, "Policy Review", "Review for completeness and compliance", "Policy approved", 4, "review", [2, 3])
            ]
        elif task_type == "strategy":
            return [
                Subgoal(1, "Market Research", "Research market conditions and requirements", "Market analysis complete", 6, "research", []),
                Subgoal(2, "Strategic Analysis", "Analyze opportunities and create framework", "Strategic framework developed", 6, "analysis", [1]),
                Subgoal(3, "Strategy Drafting", "Create comprehensive strategy document", "Strategy document complete", 8, "draft", [1]),
                Subgoal(4, "Strategy Review", "Review and refine strategy", "Strategy validated", 4, "review", [2, 3])
            ]
        elif task_type == "analysis":
            return [
                Subgoal(1, "Data Collection", "Gather relevant data and information", "Data collected", 6, "research", []),
                Subgoal(2, "Analysis", "Analyze data and identify insights", "Analysis complete", 6, "analysis", [1]),
                Subgoal(3, "Report Creation", "Create comprehensive analysis report", "Report complete", 8, "draft", [1]),
                Subgoal(4, "Report Review", "Review analysis for accuracy", "Analysis validated", 4, "review", [2, 3])
            ]
        else:  # general or creation
            return [
                Subgoal(1, "Research", "Research requirements and gather information", "Research complete", 6, "research", []),
                Subgoal(2, "Analysis", "Analyze requirements and create plan", "Plan developed", 6, "analysis", [1]),
                Subgoal(3, "Creation", "Create the deliverable", "Deliverable complete", 8, "draft", [1]),
                Subgoal(4, "Review", "Review and finalize deliverable", "Deliverable approved", 4, "review", [2, 3])
            ]
    def _classify_task_type(self, task_description: str) -> str:
        """Classify task type, preferring upstream task analysis when available."""
        inferred = str(self.task_analysis.get("task_type", "")).strip().lower() if isinstance(self.task_analysis, dict) else ""
        if inferred:
            return inferred

        desc_lower = task_description.lower()
        if any(kw in desc_lower for kw in ["policy", "guidelines", "rules", "standards", "compliance", "regulation"]):
            return "policy"
        if any(kw in desc_lower for kw in ["strategy", "plan", "marketing", "business", "go-to-market", "pricing"]):
            return "strategy"
        if any(kw in desc_lower for kw in ["analyze", "analysis", "research", "study", "benchmark", "architecture", "root cause"]):
            return "technical_analysis"
        if any(kw in desc_lower for kw in ["code", "implement", "debug", "api", "algorithm", "function", "test"]):
            return "coding"
        if any(kw in desc_lower for kw in ["paper", "report", "survey", "literature", "draft"]):
            return "research_writing"
        return "general"

    def _get_agent_role_preference(self, agent: Agent, task_description: str) -> str:
        """Get agent's role preference with simple, reliable prompt"""
        
        roles_list = "\n".join([f"- {name}: {r.description}" for name, r in self.available_roles.items()])
        role_names = ", ".join(self.available_roles.keys())

        system_prompt = f"""You are selecting your role for a team project. 

Available roles:
{roles_list}

Task: {task_description}

Choose the ONE role you're best suited for. Respond with ONLY the role name ({role_names})."""

        try:
            response = chat_completion(
                system_prompt=system_prompt,
                user_prompt="Which role do you choose?",
                model=agent.model,
                base_url=self.base_url,
                timeout=30.0,
                max_tokens=50,  # Very short response
                temperature=0.3
            )
            
            # Extract role name
            text = response.text.strip()
            for role_name in self.available_roles.keys():
                if role_name.lower() in text.lower():
                    return role_name
            
            # Fallback based on agent position
            fallback_roles = list(self.available_roles.keys())
            agent_index = ord(agent.id[-1]) - ord('A')  # AgentA=0, AgentB=1, etc.
            return fallback_roles[agent_index % len(fallback_roles)]
            
        except Exception as e:
            print(f"[Tier-1] Role selection failed for {agent.id}: {e}")
            # Fallback assignment
            fallback_roles = list(self.available_roles.keys())
            agent_index = ord(agent.id[-1]) - ord('A')
            return fallback_roles[agent_index % len(fallback_roles)]
    
    def _create_optimized_team_blueprint(self, task_description: str) -> TeamBlueprint:
        """Create team blueprint with optimal role distribution"""
        
        # Ensure role diversity
        assigned_roles = set()
        members = []
        
        # Priority order based on available roles
        priority_list = list(self.available_roles.keys())
        
        # Assign roles ensuring diversity
        for i, agent in enumerate(self.agents):
            preferred_role = self.agent_preferences.get(agent.id, priority_list[0])
            
            # If preferred role is taken, assign next available
            if preferred_role in assigned_roles:
                for role_name in priority_list:
                    if role_name not in assigned_roles:
                        preferred_role = role_name
                        break
                else:
                    # All roles assigned, use preference anyway
                    pass
            
            assigned_roles.add(preferred_role)
            role = self.available_roles[preferred_role]
            
            members.append(TeamMember(
                agent_id=agent.id,
                model=agent.model,
                role=role,
                confidence=0.9  # High confidence for predefined roles
            ))
        
        # Create simple subgoals based on task type
        subgoals = self._create_task_appropriate_subgoals(task_description)
        
        return TeamBlueprint(
            members=members,
            roles=[member.role for member in members],
            subgoals=subgoals,
            formation_turns=len(self.formation_messages)
        )
    
    def _extract_task_keywords(self, task_description: str) -> str:
        """Extract key nouns/topics from task description for dynamic subgoals."""
        # Simple extraction: skip common words and take 2-3 significant ones
        stop_words = {'a', 'an', 'the', 'for', 'and', 'with', 'create', 'write', 'build', 'design', 'about', 'of', 'to', 'in', 'on', 'at'}
        words = re.findall(r'\b\w+\b', task_description.lower())
        keywords = [w.capitalize() for w in words if w not in stop_words and len(w) > 3]
        
        # Take up to 3 keywords
        if not keywords:
            return "Task"
        return " ".join(keywords[:3])

    # NOTE: _create_task_appropriate_subgoals is defined earlier (L1040)
    # with role-aware parallel DAG generation.  Do NOT duplicate it here.
    
    def _create_single_agent_blueprint(self, task_description: str) -> TeamBlueprint:
        """Create blueprint for single agent"""
        
        # Determine roles for task anyway (for system integrity)
        self.available_roles = self._generate_custom_roles(task_description)
        
        agent = self.agents[0]
        # Assign the first (primary) role
        role_name = list(self.available_roles.keys())[0]
        role = self.available_roles[role_name]
        
        subgoals = self._create_task_appropriate_subgoals("general task")
        
        return TeamBlueprint(
            members=[TeamMember(
                agent_id=agent.id,
                model=agent.model,
                role=role,
                confidence=1.0
            )],
            roles=[role],
            subgoals=subgoals,
            formation_turns=0
        )


