"""
Convergence Monitor for AGNN

Watches multiple signals across the execution lifecycle and makes unified
decisions about when to continue, relax thresholds, force-summarize, or stop.

Replaces scattered convergence logic (PolicyController, saturation checks,
rejection counting) with a single, visible decision point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


@dataclass
class ConvergenceSignals:
    """Snapshot of all signals the monitor uses to make a decision."""
    avg_tis: float = 0.0
    tis_trend: float = 0.0          # positive = improving, negative = declining
    novelty_trend: float = 0.0      # average SD over recent window
    rejection_rate: float = 0.0     # rejected / (rejected + accepted)
    rewrite_count: int = 0
    consecutive_rejections: int = 0
    turns_completed: int = 0
    turns_budget: int = 10
    accepted_count: int = 0
    stall_turns: int = 0            # turns since last accepted message


class ConvergenceDecision:
    """The monitor's output — what should happen next."""
    CONTINUE = "continue"
    RELAX_THRESHOLD = "relax_threshold"
    FORCE_SUMMARIZE = "force_summarize"
    STOP_PHASE = "stop_phase"

    def __init__(self, action: str, reason: str, relaxation_factor: float = 1.0):
        self.action = action
        self.reason = reason
        self.relaxation_factor = relaxation_factor   # multiplier on thresholds (< 1.0 = easier)

    def __repr__(self) -> str:
        return f"ConvergenceDecision({self.action}, reason='{self.reason}')"


class ConvergenceMonitor:
    """
    Unified convergence controller for AGNN workspace execution.

    Call `evaluate()` after each turn to get a decision about what to do next.
    The orchestrator acts on the decision instead of checking scattered conditions.
    """

    def __init__(
        self,
        min_turns: int = 2,
        max_stall_turns: int = 4,
        max_consecutive_rejections: int = 3,
        novelty_floor: float = 0.08,
        tis_floor: float = 0.20,
        relaxation_step: float = 0.85,
        min_relaxation: float = 0.50,
    ):
        self.min_turns = min_turns
        self.max_stall_turns = max_stall_turns
        self.max_consecutive_rejections = max_consecutive_rejections
        self.novelty_floor = novelty_floor
        self.tis_floor = tis_floor
        self.relaxation_step = relaxation_step
        self.min_relaxation = min_relaxation

        # Internal tracking
        self._tis_history: List[float] = []
        self._sd_history: List[float] = []
        self._accepted_count: int = 0
        self._rejected_count: int = 0
        self._consecutive_rejections: int = 0
        self._rewrite_count: int = 0
        self._stall_turns: int = 0
        self._current_relaxation: float = 1.0

    def record_turn(self, tis: float, sd: float, accepted: bool, rewrote: bool = False) -> None:
        """Record metrics from a completed turn."""
        self._tis_history.append(tis)
        self._sd_history.append(sd)

        if accepted:
            self._accepted_count += 1
            self._consecutive_rejections = 0
            self._stall_turns = 0
        else:
            self._rejected_count += 1
            self._consecutive_rejections += 1
            self._stall_turns += 1

        if rewrote:
            self._rewrite_count += 1

    def evaluate(self, turns_budget: int = 10) -> ConvergenceDecision:
        """
        Evaluate all convergence signals and return a unified decision.

        Called after each turn by the orchestrator.
        """
        total_turns = len(self._tis_history)
        signals = self._compute_signals(turns_budget)

        # ── Rule 1: Not enough turns yet — always continue ──
        if total_turns < self.min_turns:
            return ConvergenceDecision(
                ConvergenceDecision.CONTINUE,
                reason=f"Warming up ({total_turns}/{self.min_turns} min turns)"
            )

        # ── Rule 2: Budget exhausted — force stop ──
        if total_turns >= turns_budget:
            return ConvergenceDecision(
                ConvergenceDecision.STOP_PHASE,
                reason=f"Turn budget exhausted ({total_turns}/{turns_budget})"
            )

        # ── Rule 3: Stalled — too many turns without an accepted message ──
        if self._stall_turns >= self.max_stall_turns:
            return ConvergenceDecision(
                ConvergenceDecision.FORCE_SUMMARIZE,
                reason=f"Stalled: {self._stall_turns} turns without accepted message"
            )

        # ── Rule 4: Excessive consecutive rejections — relax thresholds ──
        if self._consecutive_rejections >= self.max_consecutive_rejections:
            self._current_relaxation = max(
                self.min_relaxation,
                self._current_relaxation * self.relaxation_step
            )
            return ConvergenceDecision(
                ConvergenceDecision.RELAX_THRESHOLD,
                reason=f"{self._consecutive_rejections} consecutive rejections",
                relaxation_factor=self._current_relaxation,
            )

        # ── Rule 5: Novelty collapse — content is becoming repetitive ──
        if len(self._sd_history) >= 3:
            recent_sd = sum(self._sd_history[-3:]) / 3
            if recent_sd < self.novelty_floor:
                return ConvergenceDecision(
                    ConvergenceDecision.FORCE_SUMMARIZE,
                    reason=f"Novelty collapsed (avg SD={recent_sd:.3f} < {self.novelty_floor})"
                )

        # ── Rule 6: TIS declining trend — quality is getting worse ──
        if len(self._tis_history) >= 4:
            recent_avg = sum(self._tis_history[-2:]) / 2
            earlier_avg = sum(self._tis_history[-4:-2]) / 2
            if recent_avg < earlier_avg - 0.1 and recent_avg < self.tis_floor:
                return ConvergenceDecision(
                    ConvergenceDecision.STOP_PHASE,
                    reason=f"TIS declining ({earlier_avg:.3f} → {recent_avg:.3f})"
                )

        # ── Rule 7: High rejection rate mid-run — relax gently ──
        if signals.rejection_rate > 0.6 and total_turns >= 4:
            self._current_relaxation = max(
                self.min_relaxation,
                self._current_relaxation * self.relaxation_step
            )
            return ConvergenceDecision(
                ConvergenceDecision.RELAX_THRESHOLD,
                reason=f"High rejection rate ({signals.rejection_rate:.0%})",
                relaxation_factor=self._current_relaxation,
            )

        # ── Default: all healthy, continue ──
        return ConvergenceDecision(
            ConvergenceDecision.CONTINUE,
            reason="All signals healthy"
        )

    def _compute_signals(self, turns_budget: int) -> ConvergenceSignals:
        """Compute the current convergence signals snapshot."""
        total = self._accepted_count + self._rejected_count
        recent_tis = self._tis_history[-5:] if self._tis_history else [0.0]
        recent_sd = self._sd_history[-5:] if self._sd_history else [0.0]

        # TIS trend: slope of recent window
        tis_trend = 0.0
        if len(recent_tis) >= 2:
            tis_trend = recent_tis[-1] - recent_tis[0]

        return ConvergenceSignals(
            avg_tis=sum(recent_tis) / len(recent_tis),
            tis_trend=tis_trend,
            novelty_trend=sum(recent_sd) / len(recent_sd),
            rejection_rate=self._rejected_count / max(total, 1),
            rewrite_count=self._rewrite_count,
            consecutive_rejections=self._consecutive_rejections,
            turns_completed=len(self._tis_history),
            turns_budget=turns_budget,
            accepted_count=self._accepted_count,
            stall_turns=self._stall_turns,
        )

    def get_summary(self) -> Dict:
        """Return a summary dict for logging/analytics."""
        signals = self._compute_signals(0)
        return {
            "total_turns": len(self._tis_history),
            "accepted": self._accepted_count,
            "rejected": self._rejected_count,
            "rewrites": self._rewrite_count,
            "rejection_rate": round(signals.rejection_rate, 3),
            "avg_tis": round(signals.avg_tis, 3),
            "novelty_trend": round(signals.novelty_trend, 3),
            "current_relaxation": round(self._current_relaxation, 3),
        }
