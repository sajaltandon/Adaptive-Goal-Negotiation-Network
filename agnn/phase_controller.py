"""
Tier-2 Phase Controller — DAG Scheduler

Pure DAG scheduling for AGNN subgoal execution.
Each agent runs independently to completion in its own workspace;
this controller tracks which subgoals are ready, active, and done.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass

from .subgoal_decomposer import Subgoal
from .config import PhaseConfig


class PhaseController:
    """
    DAG scheduler for task subgoals.

    Tracks completed/active subgoal sets and unlocks downstream nodes
    as their dependencies are satisfied.  Content quality and turn counting
    are handled by each AgentWorkspace — this class only manages graph state.
    """

    def __init__(self, subgoals: List[Subgoal], user_prompt: str = "",
                 config: Optional[PhaseConfig] = None):
        self.subgoals = list(subgoals)
        self.user_prompt = user_prompt
        self.config = config or PhaseConfig()

        # Core DAG state
        self.completed_phase_ids: set = set()
        self.active_phase_ids: set = set()
        self.subgoal_map: Dict[int, Subgoal] = {sg.id: sg for sg in self.subgoals}

        # Lightweight per-phase metrics (populated by workspace callbacks)
        self.phase_turn_count: Dict[int, int] = {}
        self.phase_tis_scores: Dict[int, List[float]] = {}
        self.phase_eic_scores: Dict[int, List[float]] = {}
        self.phase_history: List[Dict] = []

        # GNE: agent assignment tracking (used by analytics / finalization)
        self.subgoal_assignments: Dict[int, str] = {}
        self.agent_performance: Dict[str, Dict] = {}
        self.agent_subgoal_history: Dict[str, List[int]] = {}

        # Legacy index for backwards-compatible logging
        self.current_phase_index: int = 0

        # Activate root nodes (no dependencies)
        self._update_active_phases()

    # ------------------------------------------------------------------
    # Core DAG operations
    # ------------------------------------------------------------------

    def _update_active_phases(self) -> None:
        """Activate any subgoal whose dependencies are all complete."""
        for sg in self.subgoals:
            if sg.id in self.active_phase_ids or sg.id in self.completed_phase_ids:
                continue
            deps = getattr(sg, "dependencies", [])
            if all(dep_id in self.completed_phase_ids for dep_id in deps):
                self.active_phase_ids.add(sg.id)
                self.phase_turn_count[sg.id] = 0
                self.phase_tis_scores[sg.id] = []
                self.phase_eic_scores[sg.id] = []

    def get_active_phases(self) -> List[Subgoal]:
        """Return all currently runnable subgoals (dependencies satisfied)."""
        return [self.subgoal_map[pid] for pid in self.active_phase_ids
                if pid in self.subgoal_map]

    def mark_phase_complete(self, phase_id: int) -> None:
        """Mark a subgoal as complete and unlock any newly ready successors."""
        if phase_id in self.active_phase_ids:
            self.active_phase_ids.discard(phase_id)
            self.completed_phase_ids.add(phase_id)
            self.current_phase_index += 1
            self._update_active_phases()

    def force_advance(self, phase_id: int = None) -> None:
        """Force-complete a specific subgoal (budget exceeded or emergency)."""
        if phase_id is not None and phase_id in self.active_phase_ids:
            self.mark_phase_complete(phase_id)
        elif phase_id is None and self.active_phase_ids:
            self.mark_phase_complete(next(iter(self.active_phase_ids)))

    def get_ancestor_phase_ids(self, phase_id: int) -> set:
        """Recursively collect all upstream dependency IDs for a subgoal."""
        ancestors: set = set()
        if phase_id not in self.subgoal_map:
            return ancestors

        def _traverse(pid: int) -> None:
            node = self.subgoal_map.get(pid)
            if node:
                for dep in getattr(node, "dependencies", []):
                    if dep not in ancestors:
                        ancestors.add(dep)
                        _traverse(dep)

        _traverse(phase_id)
        return ancestors

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> Optional[Subgoal]:
        """Primary active subgoal (first in set; for backwards-compat logging)."""
        if self.active_phase_ids:
            pid = next(iter(self.active_phase_ids))
            return self.subgoal_map.get(pid)
        return None

    @property
    def is_complete(self) -> bool:
        """True when every subgoal in the DAG is marked complete."""
        return len(self.completed_phase_ids) == len(self.subgoals)

    @property
    def progress_percentage(self) -> float:
        """Fraction of subgoals completed (0.0 – 1.0)."""
        if not self.subgoals:
            return 1.0
        return len(self.completed_phase_ids) / len(self.subgoals)

    # ------------------------------------------------------------------
    # Lightweight metrics recording (called by AgentWorkspace on accept)
    # ------------------------------------------------------------------

    def record_turn_metrics(self, phase_id: int, tis: float, eic: float,
                            metrics: Dict[str, float]) -> None:
        """Record per-turn TIS/EIC for a subgoal branch."""
        if phase_id in self.phase_turn_count:
            self.phase_turn_count[phase_id] += 1
            self.phase_tis_scores[phase_id].append(tis)
            self.phase_eic_scores[phase_id].append(eic)

    # ------------------------------------------------------------------
    # GNE: agent assignment + performance tracking (analytics hooks)
    # ------------------------------------------------------------------

    def set_assignment(self, subgoal_id: int, agent_id: str) -> None:
        """Record which agent was assigned to a subgoal."""
        self.subgoal_assignments[subgoal_id] = agent_id
        self.agent_subgoal_history.setdefault(agent_id, []).append(subgoal_id)

    def get_assigned_agent(self, subgoal_id: int) -> Optional[str]:
        """Return the agent assigned to a subgoal."""
        return self.subgoal_assignments.get(subgoal_id)

    def record_agent_turn(self, agent_id: str, tis: float, eic: float) -> None:
        """Update running agent performance stats."""
        perf = self.agent_performance.setdefault(agent_id, {
            "turns": 0, "total_tis": 0.0,
            "completed_subgoals": 0, "total_subgoals": 0,
        })
        perf["turns"] += 1
        perf["total_tis"] += tis
        perf["avg_tis"] = perf["total_tis"] / perf["turns"]

    def record_subgoal_completion(self, agent_id: str, subgoal_id: int,
                                  success: bool) -> None:
        """Record subgoal completion outcome for an agent."""
        perf = self.agent_performance.get(agent_id)
        if not perf:
            return
        perf["total_subgoals"] += 1
        if success:
            perf["completed_subgoals"] += 1
        total = perf["total_subgoals"]
        if total > 0:
            perf["completion_rate"] = perf["completed_subgoals"] / total

    # ------------------------------------------------------------------
    # Summary (used by _finalize_conversation in orchestrator)
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict:
        """Return a JSON-serialisable summary of DAG execution."""
        return {
            "total_phases": len(self.subgoals),
            "completed_phases": len(self.completed_phase_ids),
            "active_phases": list(self.active_phase_ids),
            "current_phase": self.current_phase.name if self.current_phase else "Complete",
            "progress": f"{int(self.progress_percentage * 100)}%",
            "phase_history": self.phase_history,
        }

    def get_transition_message(self, previous_phase: Subgoal,
                               new_phase: Optional[Subgoal]) -> str:
        """Human-readable phase transition banner."""
        if not new_phase:
            return f"[PHASE COMPLETE] {previous_phase.name} finished. All phases done!"
        turns = self.phase_turn_count.get(previous_phase.id, "?")
        return (
            f"[PHASE TRANSITION]\n"
            f"Completed: {previous_phase.name} ({turns} turns)\n"
            f"Starting:  {new_phase.name}\n"
            f"Focus:     {new_phase.description}"
        )
