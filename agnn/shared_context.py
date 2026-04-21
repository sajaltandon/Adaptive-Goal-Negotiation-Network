"""
Shared Context Board for AGNN

A lightweight shared state layer that keeps parallel workspaces aligned.
Stores only what matters for cross-workspace coordination:
  - Current task goal
  - Agreed assumptions
  - Completed subgoal outputs (structured)
  - Unresolved questions
  - Important constraints

Designed to sit on top of MessageBus, providing structured context
instead of raw broadcast summaries.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SubgoalOutput:
    """Structured output from a completed subgoal workspace."""
    subgoal_id: int
    subgoal_name: str
    objective: str
    key_findings: List[str]
    recommendation: str
    open_issues: List[str]
    confidence: float = 0.7       # 0.0-1.0 how confident the agent is


class SharedContextBoard:
    """
    Thread-safe shared state for cross-workspace alignment.

    Usage:
        board = SharedContextBoard(task_goal="Design a disaster recovery plan")

        # Workspace A finishes research:
        board.add_completed_output(SubgoalOutput(
            subgoal_id=1,
            subgoal_name="Market Research",
            objective="Identify key market players",
            key_findings=["TAM is $180B", "Top 3: Shopify, BigCommerce, WooCommerce"],
            recommendation="Focus on mid-market segment",
            open_issues=["Need pricing data for APAC region"],
            confidence=0.85,
        ))

        # Workspace B reads context before generating:
        context = board.format_context_for_agent(exclude_subgoal_id=2)
    """

    def __init__(self, task_goal: str = ""):
        self._lock = threading.Lock()
        self.task_goal: str = task_goal
        self.assumptions: List[str] = []
        self.constraints: List[str] = []
        self.completed_outputs: Dict[int, SubgoalOutput] = {}
        self.unresolved_questions: List[str] = []

    def set_task_goal(self, goal: str) -> None:
        with self._lock:
            self.task_goal = goal

    def add_assumption(self, assumption: str) -> None:
        with self._lock:
            if assumption not in self.assumptions:
                self.assumptions.append(assumption)

    def add_constraint(self, constraint: str) -> None:
        with self._lock:
            if constraint not in self.constraints:
                self.constraints.append(constraint)

    def add_question(self, question: str) -> None:
        with self._lock:
            if question not in self.unresolved_questions:
                self.unresolved_questions.append(question)

    def resolve_question(self, question: str) -> None:
        with self._lock:
            if question in self.unresolved_questions:
                self.unresolved_questions.remove(question)

    def add_completed_output(self, output: SubgoalOutput) -> None:
        with self._lock:
            self.completed_outputs[output.subgoal_id] = output

    def get_completed_findings(self, exclude_subgoal_id: Optional[int] = None) -> List[SubgoalOutput]:
        """Get all completed outputs, optionally excluding a specific subgoal."""
        with self._lock:
            return [
                out for sid, out in self.completed_outputs.items()
                if sid != exclude_subgoal_id
            ]

    def format_context_for_agent(
        self,
        exclude_subgoal_id: Optional[int] = None,
        max_findings_per_output: int = 4,
    ) -> str:
        """
        Build a compact context string suitable for injection into agent prompts.

        Returns empty string if there's nothing to share.
        """
        lines: List[str] = []

        with self._lock:
            # Task goal
            if self.task_goal:
                lines.append(f"[Task Goal]: {self.task_goal}")

            # Assumptions
            if self.assumptions:
                lines.append("[Agreed Assumptions]:")
                for a in self.assumptions[:5]:
                    lines.append(f"  • {a}")

            # Constraints
            if self.constraints:
                lines.append("[Constraints]:")
                for c in self.constraints[:5]:
                    lines.append(f"  • {c}")

            # Completed work from other subgoals
            relevant = [
                out for sid, out in self.completed_outputs.items()
                if sid != exclude_subgoal_id
            ]
            if relevant:
                lines.append("[Completed Work — build on this, do NOT repeat it]:")
                for out in relevant:
                    lines.append(f"  [{out.subgoal_name}] (confidence: {out.confidence:.0%})")
                    findings = out.key_findings[:max_findings_per_output]
                    for f in findings:
                        lines.append(f"    - {f}")
                    if out.recommendation:
                        lines.append(f"    → Recommendation: {out.recommendation}")
                    if out.open_issues:
                        for issue in out.open_issues[:2]:
                            lines.append(f"    ⚠ Open: {issue}")

            # Unresolved questions
            if self.unresolved_questions:
                lines.append("[Unresolved Questions — address if you can]:")
                for q in self.unresolved_questions[:4]:
                    lines.append(f"  ? {q}")

        return "\n".join(lines) if lines else ""

    def clear(self) -> None:
        with self._lock:
            self.assumptions.clear()
            self.constraints.clear()
            self.completed_outputs.clear()
            self.unresolved_questions.clear()
