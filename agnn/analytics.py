"""
Enhanced Analytics Module for AGNN
Provides detailed performance tracking, agent analytics, and exportable reports.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
import time
from pathlib import Path


@dataclass
class AgentPerformance:
    """Track performance metrics for a single agent"""
    agent_id: str
    role: str
    total_attempts: int = 0
    accepted: int = 0
    rejected: int = 0
    total_tis: float = 0.0
    phase_performance: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {"attempts": 0, "accepted": 0}))
    rejection_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.total_attempts if self.total_attempts > 0 else 0.0
    
    @property
    def avg_tis(self) -> float:
        return self.total_tis / self.accepted if self.accepted > 0 else 0.0
    
    def record_attempt(self, phase: str, accepted: bool, tis: float = 0.0, rejection_reason: str = None):
        """Record a message attempt"""
        self.total_attempts += 1
        self.phase_performance[phase]["attempts"] += 1
        
        if accepted:
            self.accepted += 1
            self.phase_performance[phase]["accepted"] += 1
            self.total_tis += tis
        else:
            self.rejected += 1
            if rejection_reason:
                self.rejection_reasons[rejection_reason] += 1
    
    def get_phase_rate(self, phase: str) -> float:
        """Get acceptance rate for a specific phase"""
        stats = self.phase_performance.get(phase, {"attempts": 0, "accepted": 0})
        return stats["accepted"] / stats["attempts"] if stats["attempts"] > 0 else 0.0
    
    def get_best_phase(self) -> Tuple[str, float]:
        """Get the phase where this agent performs best"""
        best_phase = None
        best_rate = 0.0
        
        for phase, stats in self.phase_performance.items():
            rate = stats["accepted"] / stats["attempts"] if stats["attempts"] > 0 else 0.0
            if rate > best_rate:
                best_rate = rate
                best_phase = phase
        
        return best_phase or "none", best_rate


@dataclass
class PhaseAnalytics:
    """Track analytics for a single phase"""
    phase_name: str
    phase_type: str
    turns: int = 0
    total_attempts: int = 0
    accepted: int = 0
    total_tis: float = 0.0
    total_eic: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    rejection_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    agent_attempts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.total_attempts if self.total_attempts > 0 else 0.0
    
    @property
    def avg_tis(self) -> float:
        return self.total_tis / self.accepted if self.accepted > 0 else 0.0
    
    @property
    def avg_eic(self) -> float:
        return self.total_eic / self.accepted if self.accepted > 0 else 0.0
    
    @property
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time
    
    def record_turn(self, agent_id: str, accepted: bool, tis: float = 0.0, eic: float = 0.0, rejection_reason: str = None):
        """Record a turn in this phase"""
        self.total_attempts += 1
        self.agent_attempts[agent_id] += 1
        
        if accepted:
            self.accepted += 1
            self.total_tis += tis
            self.total_eic += eic
        else:
            if rejection_reason:
                self.rejection_reasons[rejection_reason] += 1
    
    def complete(self):
        """Mark phase as completed"""
        self.end_time = time.time()


class SessionAnalytics:
    """Comprehensive analytics for an AGNN session"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        
        # Agent tracking
        self.agent_performance: Dict[str, AgentPerformance] = {}
        
        # Phase tracking
        self.phase_analytics: Dict[str, PhaseAnalytics] = {}
        self.current_phase: Optional[str] = None
        
        # Overall stats
        self.total_turns = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.total_rewrites = 0
        
        # Rejection analysis
        self.rejection_by_reason: Dict[str, int] = defaultdict(int)
        self.rejection_by_agent: Dict[str, int] = defaultdict(int)
        self.rejection_by_phase: Dict[str, int] = defaultdict(int)
    
    def register_agent(self, agent_id: str, role: str):
        """Register an agent for tracking"""
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = AgentPerformance(agent_id=agent_id, role=role)
    
    def start_phase(self, phase_name: str, phase_type: str):
        """Start tracking a new phase"""
        self.current_phase = phase_name
        if phase_name not in self.phase_analytics:
            self.phase_analytics[phase_name] = PhaseAnalytics(
                phase_name=phase_name,
                phase_type=phase_type
            )
    
    def end_phase(self, phase_name: str):
        """End tracking for a phase"""
        if phase_name in self.phase_analytics:
            self.phase_analytics[phase_name].complete()
    
    def record_turn(self, agent_id: str, accepted: bool, metrics: Dict = None, rejection_reasons: List[str] = None):
        """Record a conversation turn"""
        self.total_turns += 1
        
        # Extract metrics
        tis = metrics.get("TIS", 0.0) if metrics else 0.0
        eic = metrics.get("EIC", 0.0) if metrics else 0.0
        
        # Determine primary rejection reason
        rejection_reason = rejection_reasons[0] if rejection_reasons else None
        
        # Update agent performance
        if agent_id in self.agent_performance:
            phase = self.current_phase or "unknown"
            self.agent_performance[agent_id].record_attempt(
                phase=phase,
                accepted=accepted,
                tis=tis,
                rejection_reason=rejection_reason
            )
        
        # Update phase analytics
        if self.current_phase and self.current_phase in self.phase_analytics:
            self.phase_analytics[self.current_phase].record_turn(
                agent_id=agent_id,
                accepted=accepted,
                tis=tis,
                eic=eic,
                rejection_reason=rejection_reason
            )
        
        # Update overall stats
        if accepted:
            self.total_accepted += 1
        else:
            self.total_rejected += 1
            if rejection_reason:
                self.rejection_by_reason[rejection_reason] += 1
                self.rejection_by_agent[agent_id] += 1
                if self.current_phase:
                    self.rejection_by_phase[self.current_phase] += 1
    
    def get_summary(self) -> Dict:
        """Get comprehensive analytics summary"""
        duration = (self.end_time or time.time()) - self.start_time
        
        # Calculate overall acceptance rate
        total_attempts = self.total_accepted + self.total_rejected
        acceptance_rate = self.total_accepted / total_attempts if total_attempts > 0 else 0.0
        
        # Get agent rankings
        agent_rankings = sorted(
            self.agent_performance.values(),
            key=lambda a: a.acceptance_rate,
            reverse=True
        )
        
        # Get best performers by phase
        phase_best_performers = {}
        for phase_name in self.phase_analytics.keys():
            best_agent = None
            best_rate = 0.0
            for agent in self.agent_performance.values():
                rate = agent.get_phase_rate(phase_name)
                if rate > best_rate:
                    best_rate = rate
                    best_agent = agent.agent_id
            if best_agent:
                phase_best_performers[phase_name] = (best_agent, best_rate)
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return {
            "session_id": self.session_id,
            "duration_seconds": duration,
            "total_turns": self.total_turns,
            "accepted": self.total_accepted,
            "rejected": self.total_rejected,
            "acceptance_rate": acceptance_rate,
            "agent_rankings": [
                {
                    "agent_id": a.agent_id,
                    "role": a.role,
                    "acceptance_rate": a.acceptance_rate,
                    "attempts": a.total_attempts,
                    "accepted": a.accepted
                }
                for a in agent_rankings
            ],
            "phase_best_performers": phase_best_performers,
            "rejection_analysis": {
                "by_reason": dict(self.rejection_by_reason),
                "by_agent": dict(self.rejection_by_agent),
                "by_phase": dict(self.rejection_by_phase)
            },
            "recommendations": recommendations
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analytics"""
        recommendations = []
        
        # Recommend best agents for each phase
        for phase_name in self.phase_analytics.keys():
            best_agent = None
            best_rate = 0.0
            for agent in self.agent_performance.values():
                rate = agent.get_phase_rate(phase_name)
                if rate > best_rate and agent.phase_performance[phase_name]["attempts"] >= 3:
                    best_rate = rate
                    best_agent = agent
            
            if best_agent and best_rate > 0.7:
                recommendations.append(
                    f"Assign {best_agent.agent_id} ({best_agent.role}) to {phase_name} tasks ({best_rate*100:.0f}% success rate)"
                )
        
        # Identify struggling agents
        for agent in self.agent_performance.values():
            if agent.acceptance_rate < 0.4 and agent.total_attempts >= 5:
                worst_phase, worst_rate = min(
                    [(p, agent.get_phase_rate(p)) for p in agent.phase_performance.keys()],
                    key=lambda x: x[1],
                    default=(None, 0)
                )
                if worst_phase:
                    recommendations.append(
                        f"Avoid {agent.agent_id} in {worst_phase} phase ({worst_rate*100:.0f}% success rate)"
                    )
        
        # Threshold recommendations
        for phase_name, phase_stats in self.phase_analytics.items():
            if phase_stats.acceptance_rate < 0.5 and phase_stats.total_attempts >= 5:
                top_reason = max(phase_stats.rejection_reasons.items(), key=lambda x: x[1], default=(None, 0))
                if top_reason[0]:
                    recommendations.append(
                        f"Consider adjusting thresholds in {phase_name} phase (main issue: {top_reason[0]})"
                    )
        
        return recommendations
    
    def export_to_json(self, filepath: Path):
        """Export analytics to JSON file"""
        summary = self.get_summary()
        
        # Add detailed agent performance
        summary["agent_details"] = {}
        for agent_id, perf in self.agent_performance.items():
            summary["agent_details"][agent_id] = {
                "role": perf.role,
                "total_attempts": perf.total_attempts,
                "accepted": perf.accepted,
                "rejected": perf.rejected,
                "acceptance_rate": perf.acceptance_rate,
                "avg_tis": perf.avg_tis,
                "best_phase": perf.get_best_phase()[0],
                "phase_breakdown": {
                    phase: {
                        "attempts": stats["attempts"],
                        "accepted": stats["accepted"],
                        "rate": stats["accepted"] / stats["attempts"] if stats["attempts"] > 0 else 0.0
                    }
                    for phase, stats in perf.phase_performance.items()
                },
                "rejection_reasons": dict(perf.rejection_reasons)
            }
        
        # Add detailed phase analytics
        summary["phase_details"] = {}
        for phase_name, phase in self.phase_analytics.items():
            summary["phase_details"][phase_name] = {
                "phase_type": phase.phase_type,
                "turns": phase.turns,
                "total_attempts": phase.total_attempts,
                "accepted": phase.accepted,
                "acceptance_rate": phase.acceptance_rate,
                "avg_tis": phase.avg_tis,
                "avg_eic": phase.avg_eic,
                "duration_seconds": phase.duration,
                "rejection_reasons": dict(phase.rejection_reasons),
                "agent_attempts": dict(phase.agent_attempts)
            }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def export_to_text(self, filepath: Path):
        """Export human-readable summary to text file"""
        summary = self.get_summary()
        
        lines = []
        lines.append("═" * 65)
        lines.append("  AGNN SESSION ANALYTICS")
        lines.append(f"  Session: {self.session_id}")
        lines.append(f"  Duration: {summary['duration_seconds']:.1f}s")
        lines.append("═" * 65)
        lines.append("")
        
        # Overall performance
        lines.append("OVERALL PERFORMANCE:")
        lines.append(f"  Total Turns: {summary['total_turns']}")
        lines.append(f"  Accepted: {summary['accepted']} ({summary['acceptance_rate']*100:.1f}%)")
        lines.append(f"  Rejected: {summary['rejected']} ({(1-summary['acceptance_rate'])*100:.1f}%)")
        lines.append("")
        
        # Agent rankings
        lines.append("AGENT RANKINGS:")
        for i, agent in enumerate(summary['agent_rankings'], 1):
            medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else "  "
            lines.append(f"  {medal} {agent['agent_id']} ({agent['role']}): "
                        f"{agent['acceptance_rate']*100:.0f}% ({agent['accepted']}/{agent['attempts']} attempts)")
        lines.append("")
        
        # Phase best performers
        if summary['phase_best_performers']:
            lines.append("BEST PERFORMERS BY PHASE:")
            for phase, (agent_id, rate) in summary['phase_best_performers'].items():
                lines.append(f"  {phase}: {agent_id} ({rate*100:.0f}%)")
            lines.append("")
        
        # Recommendations
        if summary['recommendations']:
            lines.append("RECOMMENDATIONS:")
            for i, rec in enumerate(summary['recommendations'], 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
