"""
AGNN Configuration Module

Defines all thresholds, weights, and protocol rules for the Adaptive Goal Negotiation Network.
Includes Tier-0 (Multi-LLM Interaction Protocol) and Tier-2 (Goal Decomposition) settings.
Enhanced with Week 2 artifact detection improvements and critical fixes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import re


@dataclass(frozen=True)
class Thresholds:
    """AGNN Tier-0 Thresholds (OPTIMIZED FOR LOCAL MODELS)"""
    tau_R_min: float = 0.25      # Lowered for local models
    tau_R: float = 0.35          # Lowered for local models
    tau_N: float = 0.40          # Slightly lowered
    tau_U: float = 0.25          # Lowered for local models
    
    T_max: int = 1200            # Increased for draft phases
    T_max_hard: int = 3000       # Increased to match 2500 token draft limit
    L_max: float = 30.0          # Increased timeout tolerance
    K: int = 5
    rewrite_max_attempts: int = 1  # Reduced to prevent loops
    max_accepted_messages: int = 50  # Reduced for faster completion
    
    # Termination defaults
    w: int = 3
    epsilon_N: float = 0.15
    epsilon_U: float = 0.25
    
    # Tier-0 TIS Component Thresholds (OPTIMIZED FOR LOCAL MODELS - ACCEPTANCE RATE FIX)
    tau_TIS: float = 0.25        # Slightly higher for quality with strict role enforcement
    tau_SD: float = 0.10         # LOWERED from 0.18 - was causing 88% rejection rate
    tau_RC: float = 0.15         # LOWERED from 0.22 - minimum relevance threshold (NOT maximum!)
    tau_IS: float = 0.15         # Balanced interaction smoothness
    tau_EIC: float = 0.15        # Balanced entropy-based information contribution
    tau_St: float = 0.32         # Slightly higher for stability
    
    # Phase-Specific Threshold Overrides (for Draft/Review phases)
    # These are MORE PERMISSIVE to ensure content generation progresses
    draft_tau_RC: float = 0.08   # More permissive for draft phase
    draft_tau_SD: float = 0.05   # More permissive for draft phase
    review_tau_RC: float = 0.10  # Slightly permissive for review phase
    review_tau_SD: float = 0.06  # Slightly permissive for review phase
    
    # Rejection Counter & Progressive Relaxation
    max_rejections_before_bypass: int = 2  # Force accept after this many consecutive rejections
    threshold_relaxation_factor: float = 0.6  # Multiply thresholds by this after each rejection
    enable_rejection_bypass: bool = True  # Set to False to disable bypass mechanism
    
    # Meta-Learning Agent Selection (MLAS)
    enable_mlas: bool = True  # Enable intelligent agent selection via Thompson Sampling
    mlas_exploration_bonus: float = 1.0  # Higher = more exploration, lower = more exploitation
    
    # Role Enforcement Settings
    role_violation_penalty: float = 0.8  # Multiply TIS by this when role violations detected
    force_role_switching_threshold: int = 3  # Force role switch after N consecutive violations


@dataclass(frozen=True)
class Weights:
    # Legacy Utility weights (kept for compatibility)
    alpha: float = 0.50  # Actionability
    beta: float = 0.45   # Information Gain
    gamma: float = 0.08  # Token Cost penalty
    delta: float = 0.00  # Latency (unused)


@dataclass(frozen=True)
class TISWeights:
    """
    AGNN Tier-0 Interaction Score (TIS) Weights
    
    TIS = α·SD + β·RC + γ·IS + δ·EIC + ε·Sₜ
    
    Where:
      SD  = Semantic Distance (Novelty)
      RC  = Reciprocal Coherence (Relevance)
      IS  = Interaction Smoothness (Flow)
      EIC = Entropy-Based Information Contribution (Information Gain)
      Sₜ  = Stability Score (Conversation Stability)
    """
    alpha: float = 0.25   # Semantic Distance weight
    beta: float = 0.25    # Reciprocal Coherence weight
    gamma: float = 0.20   # Interaction Smoothness weight
    delta: float = 0.20   # Entropy Information Contribution weight
    epsilon: float = 0.10 # Stability Score weight


@dataclass(frozen=True)
class PhaseConfig:
    """
    AGNN Tier-2 Phase Controller Configuration (OPTIMIZED FOR LOCAL MODELS)
    """
    min_turns_per_phase: int = 2        # Reduced for faster progress
    max_turns_per_phase: int = 8        # Reduced to prevent getting stuck
    saturation_threshold: float = 0.35   # Lowered for local models
    info_gain_threshold: float = 0.20    # Lowered for local models
    force_transition_after: int = 8     # Increased slightly
    tis_window_size: int = 3            # Smaller window for faster detection
    enable_tier2: bool = True           # Enable Tier-2 goal decomposition
    draft_min_turns: int = 3            # Allow fast draft completion


@dataclass(frozen=True)
class ArtifactConfig:
    """Tier-2 Week 2: Artifact Validation Configuration"""
    # Quality thresholds for artifact validation (OPTIMIZED FOR BETTER DETECTION)
    quality_threshold: float = 0.3       # Overall quality required for completion (lowered from 0.4)
    completeness_threshold: float = 0.3  # Completeness required for draft completion (lowered from 0.4)
    structure_threshold: float = 0.3     # Structure quality required (lowered from 0.4)
    coherence_threshold: float = 0.25    # Coherence required (lowered from 0.3)
    readability_threshold: float = 0.3   # Readability required (lowered from 0.4)
    
    # Artifact detection settings
    enable_semantic_validation: bool = True  # Use advanced semantic validation
    enable_quality_scoring: bool = True      # Use comprehensive quality scoring
    enable_domain_validation: bool = True    # Use domain-specific validation
    
    # Fallback settings (when semantic analysis fails)
    fallback_min_sections: int = 3       # Minimum sections for basic validation
    fallback_min_words: int = 300        # Minimum words for basic validation
    
    # Debug and logging
    log_quality_scores: bool = True      # Log quality scores in debug mode
    log_validation_details: bool = False # Log detailed validation info


class ExecutionMode(Enum):
    STRICT = "strict"      # High quality thresholds, lower tolerance for failure
    BALANCED = "balanced"  # Default settings
    FAST = "fast"          # Lower thresholds, prioritizes completion speed

def get_thresholds_for_mode(mode: ExecutionMode = ExecutionMode.BALANCED) -> Thresholds:
    """Returns preset Thresholds based on execution mode."""
    if mode == ExecutionMode.STRICT:
        return Thresholds(
            tau_R_min=0.35, tau_R=0.45, tau_N=0.50, tau_U=0.35,
            tau_TIS=0.35, tau_SD=0.15, tau_RC=0.25, tau_IS=0.25, tau_EIC=0.25, tau_St=0.40,
            draft_tau_RC=0.15, draft_tau_SD=0.10,
            max_rejections_before_bypass=4,
            threshold_relaxation_factor=0.8
        )
    elif mode == ExecutionMode.FAST:
        return Thresholds(
            tau_R_min=0.15, tau_R=0.25, tau_N=0.30, tau_U=0.15,
            tau_TIS=0.15, tau_SD=0.05, tau_RC=0.08, tau_IS=0.08, tau_EIC=0.10, tau_St=0.20,
            draft_tau_RC=0.05, draft_tau_SD=0.02,
            max_rejections_before_bypass=1,
            threshold_relaxation_factor=0.4
        )
    # Balanced (default)
    return Thresholds()

def get_phase_config_for_mode(mode: ExecutionMode = ExecutionMode.BALANCED) -> PhaseConfig:
    """Returns preset PhaseConfig based on execution mode."""
    if mode == ExecutionMode.STRICT:
        return PhaseConfig(min_turns_per_phase=4, max_turns_per_phase=12, saturation_threshold=0.50, info_gain_threshold=0.30)
    elif mode == ExecutionMode.FAST:
        return PhaseConfig(min_turns_per_phase=1, max_turns_per_phase=5, saturation_threshold=0.20, info_gain_threshold=0.10)
    # Balanced (default)
    return PhaseConfig()

@dataclass(frozen=True)
class Tier2Thresholds:
    """Tier-2 Quality and Completion Thresholds"""
    artifact_completeness_min: float = 0.75
    quality_score_min: float = 0.70
    phase_completion_confidence: float = 0.80


# ----------------------------
# Phase-0 intent routing
# ----------------------------

class Intent(Enum):
    INTRODUCE = "introduce"
    INTERACT_AGENTS = "interact_agents"
    INTERACT_USER = "interact_user"
    DEFAULT = "default"
    DRAFT_PHASE = "draft_phase"


@dataclass(frozen=True)
class IntentOverrides:
    """
    Intent-specific threshold overrides.
    Overrides applied on top of base thresholds for certain intents.
    """
    tau_N: Optional[float] = None
    tau_U: Optional[float] = None
    tau_R: Optional[float] = None
    tau_R_min: Optional[float] = None


# CHANGED: INTRODUCE utility threshold was still too high at 0.45
INTENT_OVERRIDES: Dict[str, IntentOverrides] = {
    Intent.INTRODUCE: IntentOverrides(tau_N=0.05, tau_U=0.30),
    Intent.INTERACT_AGENTS: IntentOverrides(tau_N=0.55),  # Even stricter novelty for agent interactions
    Intent.INTERACT_USER: IntentOverrides(),
    Intent.DEFAULT: IntentOverrides(),
    Intent.DRAFT_PHASE: IntentOverrides(tau_N=0.35),  # Stricter novelty during drafting
}


# ----------------------------
# Protocol rules and patterns
# ----------------------------

@dataclass(frozen=True)
class ProtocolRules:
    """Protocol hygiene and validation rules"""
    forbid_speaker_labels: bool = True
    forbid_self_reference: bool = True
    forbid_task_assignment: bool = True
    max_rewrite_attempts: int = 2


# Tokens that indicate protocol violations or low-quality content
PROTOCOL_TOKENS = [
    "should we", "what next", "next?", "move forward", "initiate", 
    "data pull", "retrieve", "once complete", "once available"
]

# Regex for detecting and removing speaker labels
SPEAKER_LABEL_REGEX = re.compile(
    # Match explicit speaker labels at start of line, e.g., "AgentA:", "Assistant:", or bolded "**AgentA**:"
    r'^(?:\*\*[^*]{1,50}\*\*\s*:|(?:Agent[A-Za-z0-9]+|Assistant|System|Model|Responder|Introducer|Synthesizer)\s*:)',
    re.IGNORECASE | re.MULTILINE
)

PROTOCOL_RULES = ProtocolRules()


# ----------------------------
# Agent definitions
# ----------------------------

@dataclass(frozen=True)
class Agent:
    """AGNN Agent Definition"""
    id: str
    model: str
    system_prompt: str


def get_agents_for_intent(models: List[str], intent: Intent = Intent.DEFAULT) -> List[Agent]:
    """Create agents with appropriate system prompts for the given intent"""
    
    base_constraints = """
CRITICAL CONSTRAINTS:
- Do NOT prefix output with any name/model/agent label like 'X:' or '**X:**'. No speaker tags.
- Do NOT assign tasks to other agents or ask them to do specific work.
- Do NOT refer to yourself by name or model (avoid "I am X" or "As X").
- Focus on contributing substantive content, not coordination.
"""
    
    if intent == Intent.INTRODUCE:
        system_prompt = f"""You are participating in a multi-agent conversation. Introduce your capabilities and expertise briefly, then engage constructively with the task.

{base_constraints}

Introduce yourself by describing your strengths and how you can contribute, then participate actively in the discussion."""
        
    elif intent == Intent.INTERACT_AGENTS:
        system_prompt = f"""You are collaborating with other AI agents to solve a complex problem. Work together effectively by building on each other's contributions.

{base_constraints}

Focus on:
- Building on previous contributions
- Adding new insights and perspectives  
- Maintaining productive dialogue
- Contributing substantive content"""
        
    else:  # DEFAULT and others
        system_prompt = f"""You are an AI assistant collaborating with other agents. Contribute your expertise to help solve the given task effectively.

{base_constraints}

Provide thoughtful, substantive responses that advance the conversation toward the goal."""
    
    agents = []
    for i, model in enumerate(models):
        agent_id = f"Agent{chr(65 + i)}"  # AgentA, AgentB, AgentC, etc.
        agents.append(Agent(
            id=agent_id,
            model=model,
            system_prompt=system_prompt
        ))
    
    return agents


# ----------------------------
# Default instances
# ----------------------------

# Global configuration instances
thresholds = Thresholds()
weights = Weights()
tis_weights = TISWeights()
phase_config = PhaseConfig()
artifact_config = ArtifactConfig()
tier2_thresholds = Tier2Thresholds()
protocol_rules = ProtocolRules()
