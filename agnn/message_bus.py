"""
MessageBus — live broadcast channel for parallel agent workspaces.

Each workspace publishes a short summary of its progress after every accepted
turn. Other workspaces poll for these summaries before each turn so they can:
  - Avoid duplicating work already done by a parallel agent
  - Reference findings from a parallel workspace in their own output
  - Adjust scope when a sibling signals completion

Thread-safe: all operations acquire a single lock.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BusRequest:
    """A question posted by one workspace to be answered by another."""
    request_id:    str
    from_ws_id:    int
    from_ws_name:  str
    to_ws_id:      Optional[int]   # None = broadcast to all
    question:      str
    answer:        Optional[str]   = None
    answered_by:   Optional[int]   = None
    timestamp:     float           = field(default_factory=time.time)

    @property
    def is_answered(self) -> bool:
        return self.answer is not None


@dataclass
class BusMessage:
    workspace_id:  int
    subgoal_id:    int
    subgoal_name:  str
    phase_type:    str
    turn:          int
    summary:       str      # ≤300 chars extracted from agent output
    timestamp:     float = field(default_factory=time.time)
    status:        str = "running"   # "running" | "complete" | "failed"


class MessageBus:
    """
    Shared broadcast bus for parallel AgentWorkspace instances.

    Usage:
        bus = MessageBus()

        # inside workspace A (after an accepted turn):
        bus.publish(ws_id=1, subgoal_id=1, subgoal_name="Market Research",
                    phase_type="research", turn=2,
                    summary="Found TAM $180B, key players: Shopify, BigCommerce.")

        # inside workspace B (before building its context):
        others = bus.get_others(ws_id=2)
        # → [BusMessage(subgoal_name="Market Research", summary="Found TAM...")]
    """

    def __init__(self, max_per_workspace: int = 5):
        self._lock              = threading.Lock()
        self._messages: List[BusMessage] = []
        self._max_per_ws        = max_per_workspace
        self._completion: Dict[int, str] = {}   # subgoal_id → summary
        self._requests: List[BusRequest] = []   # two-way Q&A

    # ── publishing ────────────────────────────────────────────────────────────

    def publish(
        self,
        ws_id:        int,
        subgoal_id:   int,
        subgoal_name: str,
        phase_type:   str,
        turn:         int,
        summary:      str,
        status:       str = "running",
    ) -> None:
        msg = BusMessage(
            workspace_id=ws_id,
            subgoal_id=subgoal_id,
            subgoal_name=subgoal_name,
            phase_type=phase_type,
            turn=turn,
            summary=summary[:300],
            status=status,
        )
        with self._lock:
            # evict oldest message from this workspace if at cap
            ws_msgs = [m for m in self._messages if m.workspace_id == ws_id]
            if len(ws_msgs) >= self._max_per_ws:
                oldest = ws_msgs[0]
                self._messages.remove(oldest)
            self._messages.append(msg)

            if status == "complete":
                self._completion[subgoal_id] = summary[:300]

    def mark_complete(self, ws_id: int, subgoal_id: int,
                      subgoal_name: str, phase_type: str, summary: str) -> None:
        self.publish(ws_id, subgoal_id, subgoal_name, phase_type,
                     turn=0, summary=summary, status="complete")

    # ── reading ───────────────────────────────────────────────────────────────

    def get_others(
        self,
        ws_id:           int,
        max_age_seconds: float = 600.0,
    ) -> List[BusMessage]:
        """Return recent messages from all OTHER workspaces, newest first."""
        cutoff = time.time() - max_age_seconds
        with self._lock:
            return sorted(
                [m for m in self._messages
                 if m.workspace_id != ws_id and m.timestamp >= cutoff],
                key=lambda m: m.timestamp,
                reverse=True,
            )

    def get_completed_summaries(self, exclude_ws_id: int) -> Dict[int, str]:
        """Return {subgoal_id: summary} for all completed workspaces."""
        with self._lock:
            return {
                sid: s for sid, s in self._completion.items()
            }

    def format_digest(self, ws_id: int, max_entries: int = 4) -> str:
        """
        Build a compact human-readable digest of what parallel workspaces
        are working on right now — suitable for inclusion in agent prompts.

        Returns empty string if nothing to report.
        """
        others = self.get_others(ws_id)[:max_entries]
        if not others:
            return ""

        lines = ["[Parallel workspaces — do NOT repeat their work, build on it:]"]
        for m in others:
            status_tag = "DONE" if m.status == "complete" else f"turn {m.turn}"
            lines.append(
                f"  • [{m.subgoal_name} | {m.phase_type} | {status_tag}]: {m.summary}"
            )
        return "\n".join(lines)

    # ── two-way Q&A ───────────────────────────────────────────────────────────

    def post_request(
        self,
        from_ws_id:   int,
        from_ws_name: str,
        question:     str,
        to_ws_id:     Optional[int] = None,   # None = ask all parallel workspaces
    ) -> str:
        """
        Post a question from one workspace to another (or broadcast).
        Returns the request_id so the caller can poll for an answer.
        """
        req = BusRequest(
            request_id=str(uuid.uuid4())[:8],
            from_ws_id=from_ws_id,
            from_ws_name=from_ws_name,
            to_ws_id=to_ws_id,
            question=question[:200],
        )
        with self._lock:
            self._requests.append(req)
        return req.request_id

    def get_requests_for(self, ws_id: int) -> List[BusRequest]:
        """Return unanswered requests directed at ws_id (or broadcast)."""
        with self._lock:
            return [
                r for r in self._requests
                if not r.is_answered
                and r.from_ws_id != ws_id
                and (r.to_ws_id is None or r.to_ws_id == ws_id)
            ]

    def answer_request(self, request_id: str, answer: str, by_ws_id: int) -> bool:
        """Answer a pending request. Returns True if found and answered."""
        with self._lock:
            for req in self._requests:
                if req.request_id == request_id and not req.is_answered:
                    req.answer      = answer[:300]
                    req.answered_by = by_ws_id
                    return True
        return False

    def get_answer(self, request_id: str) -> Optional[str]:
        """Poll for the answer to a previously posted request."""
        with self._lock:
            for req in self._requests:
                if req.request_id == request_id:
                    return req.answer
        return None

    def format_requests_for(self, ws_id: int) -> str:
        """
        Build a compact prompt snippet showing pending questions this workspace
        should try to answer in its next turn.
        """
        pending = self.get_requests_for(ws_id)
        if not pending:
            return ""
        lines = ["[Questions from parallel workspaces — answer these if you can:]"]
        for r in pending[:3]:
            lines.append(f"  • [{r.from_ws_name} asks]: {r.question}")
        return "\n".join(lines)

    # ── utilities ─────────────────────────────────────────────────────────────

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()
            self._completion.clear()
            self._requests.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)
