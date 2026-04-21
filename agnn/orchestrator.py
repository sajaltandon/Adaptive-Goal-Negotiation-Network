"""
AGNN Orchestrator - Enhanced with Week 2 Artifact Detection

Main orchestration logic for the Adaptive Goal Negotiation Network.
Integrates Tier-0 (Multi-LLM Interaction Protocol) with Tier-2 (Goal Decomposition).
Enhanced with Week 2 semantic artifact detection and all critical fixes.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
import time
import json
import os
import re
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .config import (
    Thresholds, Weights, TISWeights, PhaseConfig, ArtifactConfig,
    ExecutionMode, get_thresholds_for_mode, get_phase_config_for_mode,
    get_agents_for_intent, Intent, SPEAKER_LABEL_REGEX, PROTOCOL_RULES,
    weights, tis_weights, artifact_config
)
from .llm_client import chat_completion, LLMResponse, configure_parallel_slots
from .metrics import score_message
from .rewriter import rewrite_message
from .subgoal_decomposer import SubgoalDecomposer, Subgoal
from .phase_controller import PhaseController
from .team_formation import TeamFormation, TeamBlueprint
from .analytics import SessionAnalytics
from .monitoring import RichLogger
from .storage import AgntMemory
from .synthesizer import synthesize
from .model_manager import ModelManager
from .message_bus import MessageBus
from .live_display import LiveDisplay
from .scorer import score_deliverable
from .llm_client import stream_chat_completion
from .convergence_monitor import ConvergenceMonitor, ConvergenceDecision
from .shared_context import SharedContextBoard, SubgoalOutput
from .handoff_protocol import HandoffPackage, create_handoff_package
from .tools import ToolRegistry
from .permissions import PermissionEnforcer, ToolExecutionMode



@dataclass
class AcceptedMessage:
    """Represents an accepted message in the conversation"""
    turn: int
    agent_id: str
    model: str
    content: str
    metrics: Dict[str, float]
    timestamp: float
    phase: str = "unknown"
    phase_id: int = 0  # 0 indicates global/root ledger, >0 indicates parallel branch
    role: str = ""    # Agent's role name (e.g. "Bias & Fairness Analyst") for Synthesizer attribution
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class AgentWorkspace:
    """
    Private execution context for one agent working on one subgoal.

    Each agent gets its own workspace with isolated message history and
    upstream deliverables fed in as structured input — not raw chat history.
    """
    agent: Any                                      # config.Agent
    member: Any                                     # team_formation.TeamMember
    subgoal: Any                                    # subgoal_decomposer.Subgoal
    upstream_deliverables: Dict[int, str]           # subgoal_id -> deliverable text
    private_messages: List[AcceptedMessage] = field(default_factory=list)
    deliverable: Optional[str] = None
    turn_count: int = 0
    rejection_count: int = 0
    trim_level: int = 3
    failure_count: int = 0
    status: str = "pending"     # pending | running | complete | failed
    hus_score: float = 0.0      # Handoff Utility Score (scored by downstream agent)
    revised: bool = False       # True if this workspace was re-run after a bounce-back
    last_rejection_reason: str = ""  # Populated on rejection so next turn includes a hint


class Orchestrator:
    """
    AGNN Orchestrator with enhanced Tier-2 capabilities and Week 2 artifact detection.
    
    Manages multi-agent conversations with quality gating, phase progression,
    and semantic artifact validation.
    """
    
    def __init__(
        self,
        base_url: str,
        models: List[str],
        force_agent_mode: bool = False,
        max_turns: int = 30,
        debug: bool = False,
        enable_tier2: bool = True,
        embedding_base_url: Optional[str] = None,
        event_callback: Optional[Any] = None,
        task_analysis: Optional[Dict[str, Any]] = None,
        display: Optional[Any] = None,   # LiveDisplay OR RichDashboard instance
        execution_mode: str = "balanced",
    ):
        """Initialize the orchestrator with enhanced capabilities"""
        self.base_url = base_url
        self.embedding_base_url = embedding_base_url or base_url  # Fallback to base_url if not provided
        self.models = models
        self.force_agent_mode = force_agent_mode
        self.max_turns = max_turns
        self.debug = debug
        self.enable_tier2 = enable_tier2
        self.enable_tier1 = True  # Tier-1 enabled by default
        self.event_callback = event_callback
        self.task_analysis = task_analysis or {}
        self.model_manager = ModelManager(self.base_url)
        
        # Conversation state
        self.accepted: List[AcceptedMessage] = []
        self.rejected_count = 0
        self.rewrite_count = 0
        self.termination_reason = ""
        
        # Tier-0 TIS tracking
        self.tis_history: List[float] = []
        self.ifcm: Dict[str, Dict[str, float]] = {}
        self.previous_message: Optional[str] = None
        self.previous_sender: Optional[str] = None
        
        # Tier-1 components
        self.team_blueprint: Optional[TeamBlueprint] = None
        self.agent_roles: Dict[str, str] = {}  # agent_id -> role_name
        
        # Tier-2 components
        self.subgoal_decomposer: Optional[SubgoalDecomposer] = None
        self.phase_controller: Optional[PhaseController] = None
        self.agent_phase_contexts: Dict[str, str] = {}

        # Tier-1.5: Team Plan (responsibilities + order)
        self.team_plan: Optional[Dict[str, Any]] = None
        self.team_plan_order_idx: Dict[str, int] = {}
        self.next_agent_hint: Optional[str] = None
        self.soft_role_order = {
            "research": ["Disaster Recovery Coordinator", "Application Developer", "Systems Administrator", "Network Engineer"],
            "analysis": ["Disaster Recovery Coordinator", "Application Developer", "Systems Administrator", "Network Engineer"],
            "draft": ["Application Developer", "Systems Administrator", "Network Engineer", "Disaster Recovery Coordinator"],
            "review": ["Disaster Recovery Coordinator", "Systems Administrator", "Application Developer", "Network Engineer"],
            "default": []
        }
        
        # Draft enforcement tracking
        self.draft_promise_agent: Optional[str] = None
        self.draft_promise_turn: int = 0
        self.draft_promise_count: int = 0
        
        self.agent_rejection_counters: Dict[str, int] = {}  # agent_id -> consecutive rejection count
        self.force_execution_next: bool = False

        # Tools and Execution
        self.tool_registry = ToolRegistry()
        self._autoload_mcp_servers()
        self.permission_enforcer = PermissionEnforcer(workspace_root=os.getcwd(), mode=ToolExecutionMode.AUTO)
        self.model_health: Dict[str, bool] = {}
        self.consecutive_rejections: int = 0
        self.negotiation_reopen_count: int = 0
        self.max_renegotiations: int = 2
        self.last_renegotiation_turn: int = 0
        self.renegotiation_cooldown_turns: int = 3
        # Agent reliability tracking (blacklist persistently failing agents)
        self.agent_failure_counts: Dict[str, int] = {}   # agent_id -> consecutive HTTP failures
        self.blacklisted_agents: set = set()              # agent_ids that are auto-removed from pool
        self.MAX_AGENT_FAILURES: int = 3                  # failures before blacklisting
        self.context_trim_on_400: bool = True             # trim history length on 400 before blacklist
        self.tool_outcome_memory: Dict[str, int] = {}     # model -> 400 failure count
        # Per-phase turn budget to prevent infinite loops in a single phase
        self.phase_turn_budget: Dict[str, int] = {       # max turns allowed per phase type
            "research": 12,
            "analysis": 10,
            "draft": 16,
            "review": 10,
            "general": 14,
        }
        self.phase_turn_counts: Dict[str, int] = {}      # actual turns taken per phase
        self.agent_ctx_trim_levels: Dict[str, int] = {}  # agent_id -> context trim level

        # Step-3A autonomy controls
        self.max_consecutive_accepts_per_agent: int = 2
        self.agent_consecutive_accepts: Dict[str, int] = {}
        self.phase_agent_turn_usage: Dict[str, Dict[str, int]] = {}
        self.invalid_agent_ref_re = re.compile(r'\bAgent[A-Z]\b')
        self.enable_parallel_rounds: bool = True
        self.max_accepts_per_round: int = 1
        self.enable_branch_merge: bool = True
        self.merge_every_accepts: int = 3
        self.branch_buffers: Dict[str, Dict[str, List[str]]] = {}  # phase -> agent -> snippets
        self.phase_accept_counts: Dict[str, int] = {}
        self._in_merge_checkpoint: bool = False
        self.low_tis_streak: Dict[str, int] = {}
        self.phase_start_accept_idx: Dict[str, int] = {}
        self.phase_baseline_tis: Dict[str, float] = {}
        self.last_reconcile_turn: int = 0
        self.last_verify_turn: int = 0
        self.parallel_subgoal_groups: List[List[int]] = []
        
        # Performance monitoring (NEW)
        self.agent_turn_history: Dict[str, List[Dict]] = {}  # agent_id -> [turn_data]
        self.phase_performance_tracking: Dict[str, Dict] = {}  # phase_id -> performance_data
        
        # Enhanced Analytics (NEW)
        session_id = f"agnn-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.analytics = SessionAnalytics(session_id=session_id)
        
        # Configuration objects (CRITICAL FIX: Store as instance variables)
        try:
            mode_enum = ExecutionMode(execution_mode.lower())
        except ValueError:
            print(f"Warning: Unknown execution mode '{execution_mode}', defaulting to balanced.")
            mode_enum = ExecutionMode.BALANCED
            
        self.thresholds = get_thresholds_for_mode(mode_enum)
        self.phase_config = get_phase_config_for_mode(mode_enum)
        self.weights = weights
        self.tis_weights = tis_weights
        self.artifact_config = artifact_config
        
        # Logging
        self.full_log: List[Dict[str, Any]] = []
        self.user_prompt: str = ""  # Store user prompt for filename
        self.planning_prompt: str = ""  # Normalized prompt for internal planning
        self.rejected_messages: List[Dict[str, Any]] = []  # Track rejected messages
        
        # Real-time Dashboard (Refactored to RichLogger)
        self.logger = RichLogger(debug=debug)
        self.logger.print_system_header()
            
        # Inter-session Memory ("The Brain")
        self.memory = None
        try:
            self.memory = AgntMemory()
            self.ifcm = self.memory.get_ifcm()  # Load historical influence matrix
        except Exception as e:
            print(f"Warning: Could not load memory: {e}")

        # Thread safety: lock for shared state mutations during parallel workspace execution
        self._workspace_lock = threading.Lock()
        self._active_workspace_count = 0          # how many workspaces are running in parallel right now

        # Live message bus — parallel workspaces broadcast progress to each other
        self.message_bus = MessageBus(max_per_workspace=4)

        # Live terminal display (accepts any object matching the LiveDisplay API)
        if display is not None:
            self.display = display
        else:
            self.display = LiveDisplay(use_colour=True)

        # Shared Context Board — structured cross-workspace alignment
        self.context_board = SharedContextBoard()

        # Handoff packages — structured deliverables from completed workspaces
        self.handoff_packages: Dict[int, HandoffPackage] = {}

    def _autoload_mcp_servers(self) -> None:
        """
        Auto-load MCP servers from JSON configuration files, if present.
        Supported locations (first match wins for each file):
        - <workspace>/mcp_servers.json
        - <workspace>/agnn/mcp_servers.json
        Format:
        {
          "servers": [
            {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\\\Users\\\\..."]}
          ]
        }
        """
        candidates = [
            os.path.join(os.getcwd(), "mcp_servers.json"),
            os.path.join(os.getcwd(), "agnn", "mcp_servers.json"),
        ]
        for cfg_path in candidates:
            if not os.path.exists(cfg_path):
                continue
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                servers = data.get("servers", []) if isinstance(data, dict) else []
                loaded = 0
                for server in servers:
                    if not isinstance(server, dict):
                        continue
                    # Skip servers explicitly disabled in config
                    if server.get("_disabled"):
                        name = server.get("name", "unknown")
                        print(f"[Tools] MCP '{name}' skipped (_disabled=true)")
                        continue
                    command = (server.get("command") or "").strip()
                    args = server.get("args") or []
                    if not command:
                        continue
                    if not isinstance(args, list):
                        args = [str(args)]
                    # Apply env overrides before launching the server
                    env_overrides = server.get("env", {})
                    if isinstance(env_overrides, dict):
                        for k, v in env_overrides.items():
                            if v and k not in os.environ:
                                os.environ[k] = str(v)
                    name = server.get("name", command)
                    print(f"[Tools] Loading MCP '{name}'...")
                    self.tool_registry.load_mcp_server(command, [str(a) for a in args])
                    loaded += 1
                print(f"[Tools] MCP autoload complete — {loaded} server(s) from {cfg_path}")
            except Exception as exc:
                print(f"[Tools] MCP autoload failed for {cfg_path}: {exc}")

    
    def run(self, user_prompt: str) -> List[AcceptedMessage]:
        """
        Run AGNN: negotiate roles, then drive each agent independently through
        the subgoal DAG.  Each agent owns its task, works to completion, and
        passes a clean deliverable to downstream agents.
        """
        self.user_prompt = user_prompt
        self.planning_prompt = self._normalize_planning_prompt(user_prompt)
        self.message_bus.clear()   # fresh bus for each run
        self.context_board.clear()
        self.context_board.set_task_goal(user_prompt)
        self.handoff_packages.clear()
        self._run_start_time = time.time()
        self._emit_event({"type": "run_started", "prompt": user_prompt, "timestamp": time.time()})

        # ── Launch the live dashboard immediately ─────────────────────────────
        if hasattr(self.display, 'run_start'):
            self.display.run_start(user_prompt, len(self.models))

        health = self._validate_models()
        self.model_health = dict(health)
        healthy_models = [m for m in self.models if health.get(m, False)]
        unhealthy_models = [m for m in self.models if not health.get(m, False)]

        if unhealthy_models:
            self.logger.print_error(
                f"Unhealthy models removed from run: {', '.join(unhealthy_models)}"
            )
        if healthy_models:
            self.models = healthy_models
        else:
            self.logger.print_error("No healthy models available after validation. Aborting run.")
            self.termination_reason = "no_healthy_models"
            self._emit_event({
                "type": "error",
                "timestamp": time.time(),
                "message": "No healthy models available after validation."
            })
            return []

        # Auto-detect LM Studio parallel slot capacity and unlock true concurrency
        if self.models:
            try:
                local_probe_model = next((m for m in self.models if not (m.startswith("models/gemini") or m.startswith("groq/"))), None)
                slots = configure_parallel_slots(
                    base_url=self.base_url,
                    model=local_probe_model or self.models[0],
                    max_slots=min(4, len(self.models)),
                )
                self._emit_event({
                    "type": "parallel_slots_configured",
                    "timestamp": time.time(),
                    "slots": slots,
                })
            except Exception as e:
                print(f"[Parallel Probe] Error during probe: {e} — using Semaphore(1)")

        intent = Intent.INTERACT_AGENTS if len(self.models) > 1 else Intent.DEFAULT
        agents = get_agents_for_intent(self.models, intent)
        self.agents = agents

        # Tier-2: subgoal DAG (Generate the plan FIRST)
        if self.enable_tier2:
            self._initialize_tier2_dag(self.planning_prompt or user_prompt, agents)

        # Tier-1: Role Bidding (Agents bid on the generated DAG subgoals)
        if self.enable_tier1 and len(agents) > 1:
            subgoals = self.phase_controller.subgoals if self.phase_controller else []
            self._initialize_tier1(self.planning_prompt or user_prompt, agents, subgoals=subgoals)


        # Fallback if Tier-2 failed
        if not self.enable_tier2 or not self.phase_controller:
            return self._run_fallback(user_prompt, agents)

        # Map subgoals -> agents by role-phase affinity
        assignments = self._assign_agents_to_subgoals(self.team_blueprint, self.phase_controller.subgoals)

        print("\n" + "=" * 60)
        print("  AGNN: DAG-DRIVEN WORKSPACE EXECUTION")
        print(f"  {len(self.phase_controller.subgoals)} subgoals | {len(agents)} agents")
        print("=" * 60 + "\n")

        deliverables: Dict[int, str] = {}
        workspace_map: Dict[int, AgentWorkspace] = {}

        try:
            while not self.phase_controller.is_complete:
                ready = self.phase_controller.get_active_phases()
                if not ready:
                    break

                # Print the parallel batch
                print(f"\n[DAG] Parallel batch: {len(ready)} subgoal(s)")
                for sg in ready:
                    asgn = assignments.get(sg.id, [])
                    roles = ', '.join(m.role.name for _, m in asgn if m) or "unassigned"
                    deps = getattr(sg, "dependencies", [])
                    dep_names = [self.phase_controller.subgoal_map[d].name
                                 for d in deps if d in self.phase_controller.subgoal_map]
                    print(f"      • {sg.name} ({sg.phase_type}) ← agent(s): {roles}"
                          + (f"  inputs: {dep_names}" if dep_names else ""))

                # Build workspace tasks: capture current deliverables snapshot for all
                # so each parallel workspace gets a consistent view of upstream inputs
                deliverables_snapshot = dict(deliverables)

                def _run_workspace(subgoal):
                    # Use structured handoff packages if available, else raw text
                    upstream = {}
                    for sid in getattr(subgoal, "dependencies", []):
                        if sid in self.handoff_packages:
                            upstream[sid] = self.handoff_packages[sid].to_downstream_context()
                        elif sid in deliverables_snapshot:
                            upstream[sid] = deliverables_snapshot[sid]

                    asgn = assignments.get(
                        subgoal.id,
                        [(agents[0], self.team_blueprint.members[0]
                          if self.team_blueprint else None)]
                    )
                    if len(asgn) == 1:
                        agent, member = asgn[0]
                        ws = self._run_solo_agent(agent, member, subgoal, upstream, user_prompt)
                    else:
                        ws = self._run_team_agents(asgn, subgoal, upstream, user_prompt)
                    return subgoal, ws, asgn

                # Submit ALL ready subgoals simultaneously — LM Studio serialises
                # the actual LLM calls via _INFERENCE_LOCK, but workspaces run
                # concurrently (managing their own turn loops, retries, self-critique).
                batch_results: Dict[int, tuple] = {}
                with ThreadPoolExecutor(max_workers=len(ready)) as pool:
                    futures = {pool.submit(_run_workspace, sg): sg for sg in ready}
                    for fut in as_completed(futures):
                        try:
                            subgoal, ws, asgn = fut.result()
                            batch_results[subgoal.id] = (subgoal, ws, asgn)
                        except Exception as exc:
                            sg = futures[fut]
                            print(f"[DAG] Workspace for '{sg.name}' failed: {exc}")

                # Post-process completed workspaces (sequentially — safe for DAG mutations)
                for sg_id, (subgoal, ws, asgn) in batch_results.items():
                    workspace_map[subgoal.id] = ws
                    deliverables[subgoal.id] = ws.deliverable or ""

                    # Create structured HandoffPackage for downstream workspaces
                    try:
                        lead_agent = asgn[0][0] if asgn and asgn[0][0] else agents[0]
                        pkg = create_handoff_package(
                            subgoal_id=subgoal.id,
                            subgoal_name=subgoal.name,
                            phase_type=subgoal.phase_type,
                            raw_deliverable=ws.deliverable or "",
                            accepted_messages=ws.private_messages,
                            turns_taken=ws.turn_count,
                            rejected_count=ws.rejection_count,
                            model=lead_agent.model,
                            base_url=self.base_url,
                        )
                        self.handoff_packages[subgoal.id] = pkg
                        self.context_board.add_completed_output(SubgoalOutput(
                            subgoal_id=subgoal.id,
                            subgoal_name=subgoal.name,
                            objective=subgoal.description if hasattr(subgoal, "description") else subgoal.name,
                            key_findings=pkg.what_matters_downstream[:4],
                            recommendation=pkg.suggested_focus,
                            open_issues=pkg.uncertainties[:3],
                            confidence=min(1.0, pkg.avg_tis * 1.5) if pkg.avg_tis > 0 else 0.5,
                        ))
                        self._emit_event({
                            "type": "handoff_created",
                            "timestamp": time.time(),
                            "subgoal": subgoal.name,
                            "what_was_done": pkg.what_was_done,
                            "downstream_points": len(pkg.what_matters_downstream),
                            "uncertainties": len(pkg.uncertainties),
                        })
                    except Exception as exc:
                        print(f"[Handoff] Failed to create package for {subgoal.name}: {exc}")

                    if asgn and asgn[0][0]:
                        self.phase_controller.set_assignment(subgoal.id, asgn[0][0].id)

                    # HUS scoring for upstream workspaces
                    for dep_sid in getattr(subgoal, "dependencies", []):
                        if dep_sid in workspace_map:
                            hus = self._score_handoff_quality(workspace_map[dep_sid], ws)
                            workspace_map[dep_sid].hus_score = hus
                            up_role = workspace_map[dep_sid].member.role.name \
                                if workspace_map[dep_sid].member else "unknown"
                            dn_role = asgn[0][1].role.name if asgn and asgn[0][1] else "unknown"
                            print(f"      [HUS] {up_role} -> {dn_role}: {hus:.2f}")
                            if self.memory:
                                self.memory.update_handoff_memory(
                                    up_role, dn_role, hus, ws.turn_count
                                )
                            self.phase_controller.record_subgoal_completion(
                                workspace_map[dep_sid].agent.id, dep_sid, hus >= 0.5
                            )

                    # Bounce-back
                    if len(ws.private_messages) >= 3 and ws.upstream_deliverables:
                        bounce = self._check_bounce_back(ws)
                        if bounce:
                            target_sid, feedback = bounce
                            if (target_sid in workspace_map and
                                    workspace_map[target_sid].status == "complete" and
                                    not workspace_map[target_sid].revised):
                                print(f"[BOUNCE-BACK] {subgoal.name} -> "
                                      f"{self.phase_controller.subgoal_map[target_sid].name}")
                                self._emit_event({
                                    "type": "bounce_back", "timestamp": time.time(),
                                    "from": subgoal.name,
                                    "to": self.phase_controller.subgoal_map[target_sid].name,
                                    "feedback_preview": feedback[:200],
                                })
                                target_ws = workspace_map[target_sid]
                                revised_ws = self._run_solo_agent(
                                    target_ws.agent, target_ws.member,
                                    target_ws.subgoal, target_ws.upstream_deliverables,
                                    user_prompt, revision_feedback=feedback
                                )
                                revised_ws.revised = True
                                workspace_map[target_sid] = revised_ws
                                deliverables[target_sid] = revised_ws.deliverable or ""
                                ws.upstream_deliverables[target_sid] = deliverables[target_sid]

                    # Emergent subgoal
                    emergent = self._maybe_spawn_emergent_subgoal(ws, assignments, agents)
                    if emergent:
                        self.phase_controller.subgoals.append(emergent)
                        self.phase_controller.subgoal_map[emergent.id] = emergent

                    # Mark complete → unlocks successors
                    self.phase_controller.mark_phase_complete(subgoal.id)


        except KeyboardInterrupt:
            print("\n[AGNN] Run interrupted by user.")

        print("\n" + "=" * 60)
        print("  AGNN: ALL SUBGOALS COMPLETE")
        print(f"  Messages accepted: {len(self.accepted)}")
        print("=" * 60 + "\n")

        self._finalize_conversation()
        self._emit_event({
            "type": "run_completed",
            "summary": {
                "accepted": len(self.accepted),
                "rejected": self.rejected_count,
                "rewrites": self.rewrite_count,
                "workspaces": len(workspace_map),
            },
            "timestamp": time.time(),
        })
        return self.accepted

    # ==================================================================
    # NEW WORKSPACE-BASED EXECUTION ENGINE
    # ==================================================================

    def _initialize_tier2_dag(self, user_prompt: str, agents: List) -> None:
        """
        Simplified Tier-2 init: decompose task into subgoal DAG only.
        No per-message phase tracking — that is handled by AgentWorkspace.
        """
        try:
            subgoals = []
            if self.team_blueprint and self.team_blueprint.subgoals:
                subgoals = self.team_blueprint.subgoals
                print(f"[Tier-2] Using {len(subgoals)} subgoals from team blueprint.")
            else:
                print("[Tier-2] Decomposing task into subgoals...")
                self.subgoal_decomposer = SubgoalDecomposer(
                    base_url=self.base_url,
                    model=self.models[0],
                    timeout=60.0,
                )
                roles = self.team_blueprint.roles if self.team_blueprint else []
                subgoals = self.subgoal_decomposer.decompose(user_prompt, roles=roles)

            event = {"type": "subgoals_created", "timestamp": time.time(),
                     "subgoals": [s.to_dict() for s in subgoals]}
            self.full_log.append(event)
            self._emit_event(event)
            self.logger.print_subgoals([s.to_dict() for s in subgoals])

            self.phase_controller = PhaseController(subgoals, user_prompt=user_prompt)

            # ── Tell the frontend exactly what the DAG looks like ──────────
            self._emit_event({
                "type": "subgoals_initialized",
                "timestamp": time.time(),
                "subgoals": [
                    {
                        "id": s.id,
                        "name": s.name,
                        "description": getattr(s, "description", ""),
                        "phase_type": s.phase_type,
                        "dependencies": getattr(s, "dependencies", []),
                        "status": "pending",
                    }
                    for s in subgoals
                ],
            })

            if self.phase_controller.current_phase:
                self._emit_event({
                    "type": "phase_change",
                    "timestamp": time.time(),
                    "phase": self.phase_controller.current_phase.phase_type,
                    "phase_name": self.phase_controller.current_phase.name,
                    "description": self.phase_controller.current_phase.description,
                })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.logger.print_error(f"Tier-2 DAG init failed: {e}")
            self.enable_tier2 = False

    def _assign_agents_to_subgoals(self, blueprint, subgoals: List) -> Dict[int, List]:
        """
        Map each subgoal to one or more (agent, member) pairs.

        Matching strategy (in priority order):
          1. Exact match between subgoal name and primary/secondary role names
          2. Phase-type affinity against role name keywords
          3. Round-robin fallback
        """
        agents = getattr(self, "agents", [])
        if not agents:
            # Fallback for headless tests
            try:
                from .model_manager import get_agents_for_intent, Intent
                agents = get_agents_for_intent(self.models, Intent.INTERACT_AGENTS)
            except ImportError:
                pass

        if not blueprint:
            return {sg.id: [(agents[i % len(agents)], None)] for i, sg in enumerate(subgoals)}

        agent_lookup = {m.agent_id: next((a for a in agents if a.id == m.agent_id), None)
                        for m in blueprint.members}

        assignments: Dict[int, List] = {}
        used_roles: set = set()

        for i, sg in enumerate(subgoals):
            assigned = False

            # --- Pass 1: Exact Name Match (Primary or Secondary Role) ---
            # Tier-1 explicitly generates subgoals named "{Role Name} Research"
            for member in blueprint.members:
                agent = agent_lookup.get(member.agent_id)
                if not agent:
                    continue
                
                all_roles = [member.role] + getattr(member, 'secondary_roles', [])
                for r in all_roles:
                    if r.name in sg.name or sg.name.startswith(r.name):
                        assignments[sg.id] = [(agent, member)]
                        assigned = True
                        break
                if assigned:
                    break
            
            if assigned:
                continue

            # --- Pass 2: Keyword Heuristic Fallback ---
            phase_affinity = {
                "research":  ["research", "investigat", "explor", "gather", "scout", "intel", "analys", "data", "collect", "survey", "find"],
                "analysis":  ["analys", "strateg", "architect", "planner", "synthesiz", "evaluat", "assess", "compar", "diagnos", "review",],
                "draft":     ["writ", "draft", "creat", "develop", "build", "document", "produc", "design", "implement", "generat"],
                "review":    ["review", "critic", "validat", "audit", "check", "quality", "assess", "test", "proofread", "refin", "verif"],
            }
            phase_kws   = phase_affinity.get(sg.phase_type, [])
            subgoal_kws = re.findall(r"[a-z]{4,}", (sg.name + " " + sg.description).lower())
            all_keywords = list(dict.fromkeys(phase_kws + subgoal_kws))

            matches = []
            for member in blueprint.members:
                role_lower = member.role.name.lower()
                score = sum(1 for kw in all_keywords if kw in role_lower)
                if score > 0:
                    agent = agent_lookup.get(member.agent_id)
                    if agent:
                        matches.append((score, agent, member))

            if matches:
                matches.sort(key=lambda x: -x[0])
                unused = [(a, m) for _, a, m in matches if m.role.name not in used_roles]
                if unused:
                    chosen_agent, chosen_member = unused[0]
                else:
                    _, chosen_agent, chosen_member = matches[0]
                used_roles.add(chosen_member.role.name)
                assignments[sg.id] = [(chosen_agent, chosen_member)]
            else:
                # --- Pass 3: Round Robin ---
                idx = i % len(blueprint.members)
                member = blueprint.members[idx]
                agent = agent_lookup.get(member.agent_id, agents[0] if agents else None)
                assignments[sg.id] = [(agent, member)]

        # Upgrade team-convergence nodes (2+ deps) to multi-agent
        for sg in subgoals:
            deps = getattr(sg, "dependencies", [])
            if len(deps) >= 2 and sg.id in assignments:
                current = assignments[sg.id]
                current_ids = {a.id for a, _ in current if a}
                for member in blueprint.members:
                    if len(current) >= min(len(deps), len(blueprint.members)):
                        break
                    agent = agent_lookup.get(member.agent_id)
                    if agent and agent.id not in current_ids:
                        current.append((agent, member))
                        current_ids.add(agent.id)

        return assignments

    def _build_agent_context(self, ws: AgentWorkspace, user_prompt: str) -> Dict[str, str]:
        """
        Build focused, workspace-isolated context for a solo agent.
        Upstream work arrives as clean structured deliverables — not raw chat history.
        """
        agent = ws.agent
        member = ws.member
        subgoal = ws.subgoal

        base_system = agent.system_prompt if agent else ""

        if member:
            role_prompt = (
                f"\n[MANDATORY ROLE: {member.role.name}]\n"
                f"{member.role.description}\n\n"
                f"RESPONSIBILITIES:\n"
                + "\n".join(f"- {r}" for r in member.role.responsibilities)
                + f"\n\n{member.role.system_prompt_addition}\n\n"
                "You are working INDEPENDENTLY on your assigned task. "
                "Produce your full deliverable — do not wait for others."
            )
            base_system += role_prompt

            # ── Role Continuity (item 10) ───────────────────────────────────
            # Every 3 turns, re-inject a condensed role reminder directly into
            # the user message to counter gradual role drift in longer workspaces.
            ROLE_REINFORCE_EVERY = 3
            if ws.turn_count > 0 and ws.turn_count % ROLE_REINFORCE_EVERY == 0:
                responsibilities_short = "; ".join(member.role.responsibilities[:3])
                base_system += (
                    f"\n\n⚠ ROLE REMINDER (turn {ws.turn_count}): "
                    f"You MUST stay in character as [{member.role.name}]. "
                    f"Your core responsibilities: {responsibilities_short}. "
                    f"If your last response drifted from this role, immediately re-anchor."
                )

        base_system += (
            f"\n\n[YOUR TASK: {subgoal.name}]\n"
            f"Type: {subgoal.phase_type}\n"
            f"Goal: {subgoal.description}\n"
            f"Completion criteria: {subgoal.completion_criteria}\n"
        )

        # Upstream deliverables as structured handoff blocks
        upstream_block = ""
        for sid, text in ws.upstream_deliverables.items():
            sg = self.phase_controller.subgoal_map.get(sid) if self.phase_controller else None
            sg_name = sg.name if sg else f"Subgoal {sid}"
            upstream_block += f"\n\n### Input from {sg_name}:\n{text[:2000]}"

        # Private work history (only this agent's messages)
        recent = ws.private_messages[-ws.trim_level:] if ws.private_messages else []
        history_lines = []
        for m in recent:
            if m.tool_calls:
                for tc in m.tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                    history_lines.append(f"[Turn {m.turn} Tool Call]: {name}({args})")
            if m.content.strip():
                history_lines.append(f"[Turn {m.turn}]: {m.content[:400]}")
        history = "\n".join(history_lines)

        # On subsequent turns, summarise what is ALREADY covered so the agent
        # knows exactly what NOT to repeat — and what gap still needs filling.
        already_covered = ""
        advance_instruction = "Produce concrete, structured content."
        if ws.turn_count > 0 and ws.private_messages:
            topics_done = []
            for m in ws.private_messages:
                # Extract first heading or first sentence as a bullet
                first_line = m.content.strip().split("\n")[0][:80].strip("# ").strip()
                if first_line:
                    topics_done.append(f"- {first_line}")
            if topics_done:
                already_covered = (
                    "\n\nAlready covered in previous turns (DO NOT REPEAT OR REPHRASE):\n"
                    + "\n".join(topics_done[:6])
                )
            advance_instruction = (
                "IMPORTANT: Do NOT repeat, rephrase, or summarise what you already wrote above.\n"
                "Instead, identify the single most important MISSING piece for this task and write ONLY that — "
                "new specific details, numbers, examples, or sections not yet covered.\n"
                "If everything is complete, write: COMPLETE: <one sentence summary of your deliverable>"
            )

        # Live digest from parallel workspaces running at the same time.
        bus_digest    = self.message_bus.format_digest(ws_id=ws.subgoal.id, max_entries=4)
        # Pending questions other workspaces posted to this one
        bus_requests  = self.message_bus.format_requests_for(ws_id=ws.subgoal.id)

        # Shared Context Board — structured alignment from completed workspaces
        board_context = self.context_board.format_context_for_agent(
            exclude_subgoal_id=ws.subgoal.id
        )

        user = (
            f"Task: {user_prompt}\n\n"
            f"Your assignment: {subgoal.description}\n"
            f"Completion criteria: {subgoal.completion_criteria}"
            + upstream_block
            + (f"\n\n{board_context}" if board_context else "")
            + (f"\n\n{bus_digest}"   if bus_digest   else "")
            + (f"\n\n{bus_requests}" if bus_requests  else "")
            + "\n\nYour work so far:\n"
            + (history if history else "Starting your work now.")
            + already_covered
            + (
                f"\n\n⚠ FEEDBACK FROM QUALITY GATE (your last response was rejected):\n"
                f"{ws.last_rejection_reason}\n"
                f"Fix this issue in your next response before continuing."
                if ws.last_rejection_reason else ""
            )
            + "\n\nExecution constraints:"
            + "\n- Environment is Windows; use Windows-compatible commands/tools only."
            + "\n- Do not suggest Linux-specific paths/commands like /proc, ls -l, grep."
            + "\n- For numeric claims, label each as [Verified], [Estimated], or [Assumption]."
            + f"\n\n{advance_instruction}"
        )

        return {"system_prompt": base_system, "user_prompt": user}

    def _build_team_context(self, agent, member, ws: AgentWorkspace,
                            user_prompt: str) -> Dict[str, str]:
        """Context for a team-collaboration workspace (multiple agents on one node)."""
        role_name = member.role.name if member else "Agent"
        base_system = (
            (agent.system_prompt if agent else "")
            + f"\n[ROLE: {role_name}]\n"
            + (member.role.system_prompt_addition if member else "")
            + f"\n\n[TEAM TASK: {ws.subgoal.name}]\n"
            f"Collaborate to: {ws.subgoal.description}"
        )

        upstream_block = ""
        for sid, text in ws.upstream_deliverables.items():
            sg = self.phase_controller.subgoal_map.get(sid) if self.phase_controller else None
            sg_name = sg.name if sg else f"Subgoal {sid}"
            upstream_block += f"\n\n### Input from {sg_name}:\n{text[:1500]}"

        recent = ws.private_messages[-4:] if ws.private_messages else []
        history = "\n".join(
            f"{m.agent_id} ({self.agent_roles.get(m.agent_id, '?')}): {m.content[:350]}"
            for m in recent
        )

        user = (
            f"Task: {user_prompt}\n\n"
            f"Team goal: {ws.subgoal.description}\n"
            f"Criteria: {ws.subgoal.completion_criteria}"
            + upstream_block
            + "\n\nTeam discussion so far:\n"
            + (history if history else "Discussion starting now.")
            + f"\n\nContribute as {role_name}. Build on what others have said."
        )

        return {"system_prompt": base_system, "user_prompt": user}

    def _process_workspace_response(self, agent, member, response: LLMResponse,
                                    ws: AgentWorkspace, turn: int) -> bool:
        """
        TIS gate applied within a workspace context.
        Reads from workspace private messages instead of the global accepted list.
        """
        raw_text = response.text
        phase_type = ws.subgoal.phase_type
        cleaned = self._sanitize_message(raw_text)

        if self._has_invalid_agent_reference(cleaned, agent.id):
            ws.rejection_count += 1
            ws.last_rejection_reason = "Your response referenced another agent by internal codename (AgentA/B/C/D). Speak only as yourself and never mention other agents by codename."
            return False

        if self._is_repetitive_in_workspace(cleaned, ws.private_messages):
            ws.rejection_count += 1
            ws.last_rejection_reason = "Your response was too similar to content you already wrote in a previous turn. Do NOT repeat or rephrase previous content. Add genuinely new information, data, or analysis."
            return False

        context_messages = [m.content for m in ws.private_messages[-3:]]
        accepted_history = [m.content for m in ws.private_messages]
        prev_msg = ws.private_messages[-1].content if ws.private_messages else None
        tis_hist = [m.metrics.get("TIS", 0.0) for m in ws.private_messages[-10:]]

        metrics = score_message(
            candidate_text=cleaned,
            context_messages=context_messages,
            accepted_history=accepted_history,
            previous_message=prev_msg,
            tis_history=tis_hist,
            latency_seconds=response.latency_seconds,
            model=agent.model,
            base_url=self.embedding_base_url,
        )

        dynamic_thresholds = self._get_dynamic_thresholds_for_workspace(agent.id, ws)

        force_accept = (ws.rejection_count >= self.thresholds.max_rejections_before_bypass
                        and self.thresholds.enable_rejection_bypass)

        if force_accept:
            accept, reasons = True, [f"Force-accepted after {ws.rejection_count} rejections"]
        else:
            accept, reasons = self._gate(metrics, dynamic_thresholds, [])

        if accept:
            accepted_msg = AcceptedMessage(
                turn=turn,
                agent_id=agent.id,
                model=agent.model,
                content=cleaned,
                metrics=metrics,
                timestamp=time.time(),
                phase=phase_type,
                phase_id=ws.subgoal.id,
                role=member.role.name if member else "",
            )
            # Workspace-local state — no lock needed (each workspace is single-threaded)
            ws.private_messages.append(accepted_msg)
            ws.rejection_count = 0
            ws.last_rejection_reason = ""  # Clear the hint — agent corrected itself

            # Shared global state — must be lock-protected for parallel workspaces
            with self._workspace_lock:
                self.accepted.append(accepted_msg)
                self.tis_history.append(metrics.get("TIS", 0.0))
                self._update_ifcm(agent.id, self.previous_sender,
                                  metrics.get("TIS", 0.0), ws.subgoal.id)
                self.previous_message = cleaned
                self.previous_sender = agent.id

            self.analytics.record_turn(agent_id=agent.id, accepted=True,
                                       metrics=metrics, rejection_reasons=None)
            self._track_agent_performance(agent.id, metrics, turn, ws.subgoal.id)

            tis_val = float(metrics.get("TIS", 0.0))
            self.low_tis_streak[agent.id] = (
                self.low_tis_streak.get(agent.id, 0) + 1
                if tis_val < max(0.64, self.thresholds.tau_TIS)
                else 0
            )

            if self.phase_controller and ws.subgoal.id in self.phase_controller.phase_turn_count:
                self.phase_controller.record_turn_metrics(
                    ws.subgoal.id, metrics.get("TIS", 0.0), metrics.get("EIC", 0.0), metrics
                )
                self.phase_controller.record_agent_turn(
                    agent.id, metrics.get("TIS", 0.0), metrics.get("EIC", 0.0)
                )

            role = member.role.name if member else self.agent_roles.get(agent.id, "Agent")
            self.logger.print_chat_turn(agent.id, role, cleaned, metrics, status="ACCEPTED")
            self._emit_event({
                "type": "message", "timestamp": time.time(), "decision": "ACCEPT",
                "agent_id": agent.id, "agent_role": role,
                "content": cleaned, "phase": phase_type, "metrics": metrics,
            })

            # Broadcast a summary of this turn to the message bus so parallel
            # workspaces can see what this agent found and avoid duplication.
            summary_words = cleaned.split()
            bus_summary = " ".join(summary_words[:50]) + ("..." if len(summary_words) > 50 else "")
            self.message_bus.publish(
                ws_id=ws.subgoal.id,
                subgoal_id=ws.subgoal.id,
                subgoal_name=ws.subgoal.name,
                phase_type=phase_type,
                turn=turn,
                summary=bus_summary,
                status="running",
            )
            # Live display turn summary (parallel-mode: brief one-liner)
            self.display.turn_accepted(
                ws_name=ws.subgoal.name[:30],
                phase=phase_type,
                turn=turn,
                snippet=cleaned,
                tis=float(metrics.get("TIS", 0.0)),
            )
            return True

        else:
            # Workspace-local rejection count — no lock needed
            ws.rejection_count += 1
            # Global counter — lock-protected
            with self._workspace_lock:
                self.rejected_count += 1
                self.tis_history.append(metrics.get("TIS", 0.0))

            self._emit_event({
                "type": "message", "timestamp": time.time(), "decision": "REJECT",
                "agent_id": agent.id, "agent_role": member.role.name if member else "?",
                "content": cleaned, "phase": phase_type,
                "reasons": reasons, "metrics": metrics,
            })

            rewritten = rewrite_message(
                original_message=raw_text,
                rejection_reasons=reasons,
                context_messages=context_messages,
                model=agent.model,
                base_url=self.base_url,
            )
            if rewritten != raw_text:
                with self._workspace_lock:
                    self.rewrite_count += 1
                rr = LLMResponse(text=rewritten, model=agent.model,
                                 tokens_used=len(rewritten.split()), latency_seconds=0.0)
                return self._process_workspace_response(agent, member, rr, ws, turn)
            return False

    def _is_repetitive_in_workspace(self, text: str,
                                     workspace_messages: List[AcceptedMessage]) -> bool:
        """N-gram overlap check against workspace-private history (not global)."""
        if not workspace_messages or len(text.split()) < 20:
            return False
        text_words = set(text.lower().split())
        for msg in workspace_messages[-3:]:
            existing_words = set(msg.content.lower().split())
            if text_words and len(text_words & existing_words) / len(text_words) > 0.72:
                return True
        return False

    def _get_dynamic_thresholds_for_workspace(self, agent_id: str, ws: AgentWorkspace):
        """Dynamic TIS thresholds scoped to a workspace's rejection count."""
        phase_type = ws.subgoal.phase_type
        base = self.thresholds

        if phase_type == "draft":
            tau_RC, tau_SD = base.draft_tau_RC, base.draft_tau_SD
        elif phase_type == "review":
            tau_RC, tau_SD = base.review_tau_RC, base.review_tau_SD
        else:
            tau_RC, tau_SD = base.tau_RC, base.tau_SD

        if ws.rejection_count > 0 and base.enable_rejection_bypass:
            factor = base.threshold_relaxation_factor ** ws.rejection_count
            tau_RC *= factor
            tau_SD *= factor

        from dataclasses import replace as dc_replace
        return dc_replace(base, tau_RC=tau_RC, tau_SD=tau_SD)

    def _agent_task_complete(self, ws: AgentWorkspace) -> bool:
        """
        Decide whether a solo agent has finished its task.
        Uses TIS stability, content signals, and minimum turn requirements.
        """
        if not ws.private_messages:
            return False

        # Agent explicitly declared completion
        last_content = ws.private_messages[-1].content.strip()
        if last_content.upper().startswith("COMPLETE:"):
            return True

        min_turns = {"research": 4, "analysis": 3, "draft": 6, "review": 3}.get(
            ws.subgoal.phase_type, 4
        )
        if ws.turn_count < min_turns:
            return False

        # TIS plateau = topic saturated
        if len(ws.private_messages) >= 4:
            recent_tis = [m.metrics.get("TIS", 0.0) for m in ws.private_messages[-4:]]
            avg = sum(recent_tis) / len(recent_tis)
            var = sum((t - avg) ** 2 for t in recent_tis) / len(recent_tis)
            if avg >= 0.68 and var < 0.012:
                return True

        content = "\n".join(m.content for m in ws.private_messages)
        wc      = len(content.split())
        pt      = ws.subgoal.phase_type
        cl      = content.lower()

        # Phase-specific completion signals
        if pt == "research":
            return wc >= 300 and any(
                w in cl for w in ["findings", "summary", "conclusion", "identified",
                                  "requirements", "gathered", "collected"]
            )
        if pt == "analysis":
            return wc >= 400 and bool(re.search(r"^##|^\d+\.|^[-*] ", content, re.MULTILINE))
        if pt == "draft":
            sections = len(re.findall(r"^##\s+", content, re.MULTILINE))
            return wc >= 600 and sections >= 2
        if pt == "review":
            return wc >= 200 and any(
                w in cl for w in ["recommend", "suggest", "improve", "final",
                                  "complete", "approved", "verified"]
            )
        # Unknown / custom phase type: done when there's enough structured content
        has_structure = bool(re.search(r"^##|^\d+\.|^[-*] ", content, re.MULTILINE))
        return wc >= 300 and has_structure

    def _extract_deliverable(self, ws: AgentWorkspace) -> str:
        """Extract the best deliverable text from a finished workspace."""
        if not ws.private_messages:
            return f"[{ws.subgoal.name}: No content produced]"

        if ws.subgoal.phase_type == "draft":
            return "\n\n".join(m.content for m in ws.private_messages)

        sorted_msgs = sorted(ws.private_messages,
                             key=lambda m: m.metrics.get("TIS", 0.0), reverse=True)
        top = sorted(sorted_msgs[:min(4, len(sorted_msgs))], key=lambda m: m.turn)
        return "\n\n".join(m.content for m in top)

    def _deliberate_before_handoff(self, ws: AgentWorkspace, user_prompt: str) -> str:
        """
        Self-critique pass: agent reviews its own deliverable before passing it on.
        Adds missing content if found; confirms completeness otherwise.
        """
        draft = ws.deliverable or ""
        if not draft.strip() or ws.subgoal.phase_type == "review":
            return draft

        critique_prompt = (
            f"You just completed: {ws.subgoal.name}\n\n"
            f"Your deliverable:\n{draft[:2000]}\n\n"
            f"Completion criteria: {ws.subgoal.completion_criteria}\n\n"
            "Quick self-review:\n"
            "1. Does it fully satisfy the criteria?\n"
            "2. What is the single most important gap, if any?\n"
            "3. If there is a gap — add it now in 2-4 sentences.\n"
            "   If complete — respond ONLY with: COMPLETE: <one-sentence summary>"
        )
        try:
            resp = chat_completion(
                system_prompt=f"You are {ws.member.role.name if ws.member else 'the agent'}. "
                              "Review and improve your own work.",
                user_prompt=critique_prompt,
                model=ws.agent.model,
                base_url=self.base_url,
                timeout=60.0,
                max_tokens=500,
                temperature=0.3,
            )
            text = resp.text.strip()
            if text.upper().startswith("COMPLETE:"):
                return draft
            self._emit_event({
                "type": "self_critique", "timestamp": time.time(),
                "agent_id": ws.agent.id, "subgoal": ws.subgoal.name,
                "addition_preview": text[:150],
            })
            return draft + "\n\n### Self-Review Addition:\n" + text
        except Exception as e:
            print(f"[Self-Critique] {ws.agent.id}: {e}")
            return draft

    def _run_solo_agent(self, agent, member, subgoal, upstream: Dict[int, str],
                        user_prompt: str, revision_feedback: str = "") -> AgentWorkspace:
        """
        Run one agent independently on its subgoal until the task is done
        or the turn budget is exhausted.  Includes hot-swap on repeated failures
        and a self-critique pass before finalising the deliverable.
        """
        ws = AgentWorkspace(agent=agent, member=member, subgoal=subgoal,
                            upstream_deliverables=dict(upstream))
        ws.status = "running"

        if revision_feedback:
            ws.private_messages.append(AcceptedMessage(
                turn=0, agent_id="system", model="system",
                content=f"[REVISION REQUEST]:\n{revision_feedback[:600]}",
                metrics={"TIS": 0.7}, timestamp=time.time(),
                phase=subgoal.phase_type, phase_id=subgoal.id,
            ))

        max_turns   = self._get_adaptive_turn_budget(subgoal, member.role.name if member else "")
        failure_count = 0
        ws_label    = subgoal.name[:30]
        t_ws_start  = time.time()

        # Per-workspace convergence monitor
        convergence = ConvergenceMonitor(
            min_turns=2,
            max_stall_turns=4,
            max_consecutive_rejections=self.thresholds.max_rejections_before_bypass,
        )

        with self._workspace_lock:
            self._active_workspace_count += 1
        # Only stream tokens when this is the sole active workspace — multiple parallel
        # workspaces writing tokens at the same time garbles the terminal.
        streaming_enabled = self._active_workspace_count == 1

        self.display.workspace_start(ws_label, subgoal.phase_type, subgoal.description)
        msg = f"[{member.role.name if member else agent.id}] Starting: {subgoal.name}"
        if hasattr(self, 'display') and hasattr(self.display, 'bus_event'):
            self.display.bus_event("SYSTEM", "DAG", msg)
        else:
            self.logger.log_system(msg)
        self._emit_event({
            "type": "agent_task_start", "timestamp": time.time(),
            "agent_id": agent.id, "role": member.role.name if member else "?",
            "subgoal": subgoal.name, "phase_type": subgoal.phase_type,
            "max_turns": max_turns,
        })
        # ── Tell frontend this workspace is now live ───────────────────────
        self._emit_event({
            "type": "workspace_started",
            "timestamp": time.time(),
            "agent_id": agent.id,
            "model": agent.model,
            "role": member.role.name if member else agent.id,
            "subgoal": subgoal.name,
            "phase_type": subgoal.phase_type,
            "max_turns": max_turns,
        })

        while ws.turn_count < max_turns:
            ctx         = self._build_agent_context(ws, user_prompt)
            next_turn   = ws.turn_count + 1
            token_cb    = self.display.make_token_cb(ws_label, turn=next_turn) if streaming_enabled else None
            tool_schemas = self.tool_registry.get_schemas(
                model=agent.model,
                failure_level=failure_count,
                task_hint=ctx["user_prompt"],
                phase_type=subgoal.phase_type,
            )
            try:
                if streaming_enabled:
                    response = stream_chat_completion(
                        system_prompt=ctx["system_prompt"],
                        user_prompt=ctx["user_prompt"],
                        model=self._select_cost_aware_model(agent.model, subgoal.phase_type),
                        base_url=self.base_url,
                        timeout=120.0,
                        max_tokens=self._get_max_tokens_for_phase(agent.model),
                        temperature=0.7,
                        on_token=token_cb,
                        tools=tool_schemas,
                    )
                    self.display.stream_end(ws_label)
                else:
                    response = chat_completion(
                        system_prompt=ctx["system_prompt"],
                        user_prompt=ctx["user_prompt"],
                        model=self._select_cost_aware_model(agent.model, subgoal.phase_type),
                        base_url=self.base_url,
                        timeout=120.0,
                        max_tokens=self._get_max_tokens_for_phase(agent.model),
                        temperature=0.7,
                        tools=tool_schemas,
                    )
                failure_count = 0

            except Exception as e:
                err = str(e)
                # Only count HTTP/network errors toward the hot-swap threshold.
                # AttributeError / code bugs should surface immediately, not trigger
                # an infinite model-swap death loop.
                is_network_error = any(x in err for x in ("HTTP", "timed out", "Connection", "Request failed"))
                if is_network_error:
                    failure_count += 1
                    if self.debug:
                        print(f"  [{member.role.name if member else agent.id}] LLM error #{failure_count}: {e}")
                    # Report failure to ModelManager for sticky blacklisting
                    self.model_manager.report_failure(agent.model)
                else:
                    # Hard code bug — raise immediately so we can fix it
                    ws.status = "failed"
                    raise

                if "400" in err:
                    ws.trim_level = max(1, ws.trim_level - 1)
                    self.tool_outcome_memory[agent.model] = self.tool_outcome_memory.get(agent.model, 0) + 1
                    print(f"  [Memory] Tool failure logged for {agent.model}. Triggering Adaptive Surface Reduction.")

                if failure_count >= self.MAX_AGENT_FAILURES:
                    # Failure Escalation: If we failed multiple times, try cloud planner or fallback
                    if "400" in err and len(self.models) > 1:
                        # Escalation: try switching to a cloud model dynamically
                        for m in self.models:
                            if "70b" in m or "pro" in m:
                                print(f"  [Escalation] Local model {agent.model} failed. Escalating to {m}.")
                                agent.model = m
                                break
                    
                    # suggest_replacement_model already excludes blacklisted models
                    backup = self.model_manager.suggest_replacement_model(agent.model)
                    if backup and backup != agent.model:
                        old_model = agent.model
                        self.model_manager.unload_model(old_model)
                        if self.model_manager.load_model(backup):
                            object.__setattr__(agent, "model", backup)
                            failure_count = 0
                            print(f"  [Hot-Swap] {agent.id}: {old_model} \u2192 {backup}")
                            # ── Tell the frontend about the hot-swap ──────────────
                            self._emit_event({
                                "type": "hot_swap",
                                "timestamp": time.time(),
                                "agent_id": agent.id,
                                "old_model": old_model,
                                "new_model": backup,
                                "subgoal": subgoal.name,
                            })
                            continue
                    elif self.model_manager.is_blacklisted(agent.model):
                        # Model is blacklisted and no clean replacement found — give up on this branch
                        print(f"  [Hot-Swap] {agent.id}: no eligible replacement — terminating branch.")
                    ws.status = "failed"
                    break
                continue

            if getattr(response, "tool_calls", None):
                ws.turn_count += 1
                # Record the assistant's tool call request
                ws.private_messages.append(AcceptedMessage(
                    turn=ws.turn_count,
                    agent_id=agent.id,
                    model=agent.model,
                    content=response.text or "",
                    metrics={"TIS": 1.0},
                    timestamp=time.time(),
                    phase=subgoal.phase_type,
                    phase_id=subgoal.id,
                    role=member.role.name if member else agent.id,
                    tool_calls=response.tool_calls
                ))
                
                # Execute the tools
                for tc in response.tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name")
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except Exception:
                        args = {}
                    
                    if hasattr(self, 'display') and hasattr(self.display, 'tool_execution'):
                        self.display.tool_execution(ws.subgoal.name, name)
                    else:
                        self.logger.log_system(f"[{agent.id}] Executing Tool: {name}")
                        
                    result = self.tool_registry.execute(name, args, self.permission_enforcer)
                    
                    ws.turn_count += 1
                    ws.private_messages.append(AcceptedMessage(
                        turn=ws.turn_count,
                        agent_id="system",
                        model="tool",
                        content=f"Tool '{name}' Output:\n{result}",
                        metrics={"TIS": 1.0},
                        timestamp=time.time(),
                        phase=subgoal.phase_type,
                        phase_id=subgoal.id,
                        role="system"
                    ))
                    # Also mirror tool activity into global accepted stream so runs with
                    # heavy tool usage can still synthesize meaningful final outputs.
                    self.accepted.append(AcceptedMessage(
                        turn=ws.turn_count,
                        agent_id=agent.id,
                        model=agent.model,
                        content=f"[TOOL] {name}({args}) -> {str(result)[:600]}",
                        metrics={"TIS": 0.7},
                        timestamp=time.time(),
                        phase=subgoal.phase_type,
                        phase_id=subgoal.id,
                        role=member.role.name if member else agent.id,
                    ))
                continue

            ws.turn_count += 1
            accepted = self._process_workspace_response(agent, member, response, ws, ws.turn_count)

            # Record turn in convergence monitor
            metrics = ws.private_messages[-1].metrics if ws.private_messages else {}
            convergence.record_turn(
                tis=float(metrics.get("TIS", 0.0)),
                sd=float(metrics.get("SD", 0.0)),
                accepted=accepted,
            )

            # Evaluate convergence — unified decision point
            decision = convergence.evaluate(turns_budget=max_turns)
            if decision.action == ConvergenceDecision.STOP_PHASE:
                self.logger.log_system(f"[Convergence] {ws_label}: {decision.reason} — stopping.")
                break
            elif decision.action == ConvergenceDecision.FORCE_SUMMARIZE:
                self.logger.log_system(f"[Convergence] {ws_label}: {decision.reason} — force summarizing.")
                break
            elif decision.action == ConvergenceDecision.RELAX_THRESHOLD:
                self.logger.log_system(f"[Convergence] {ws_label}: relaxing thresholds (x{decision.relaxation_factor:.2f})")

            if accepted and self._agent_task_complete(ws):
                break

        ws.deliverable = self._extract_deliverable(ws)
        ws.deliverable = self._deliberate_before_handoff(ws, user_prompt)
        ws.status = "complete"

        with self._workspace_lock:
            self._active_workspace_count = max(0, self._active_workspace_count - 1)

        # Broadcast completion so parallel workspaces know this task is done
        self.message_bus.mark_complete(
            ws_id=ws.subgoal.id,
            subgoal_id=ws.subgoal.id,
            subgoal_name=subgoal.name,
            phase_type=subgoal.phase_type,
            summary=(ws.deliverable or "")[:200],
        )
        self.display.workspace_done(
            ws_label, turns=ws.turn_count,
            elapsed=time.time() - t_ws_start,
            hus=ws.hus_score if ws.hus_score > 0 else 0.0,
        )

        self._emit_event({
            "type": "agent_task_complete", "timestamp": time.time(),
            "agent_id": agent.id, "role": member.role.name if member else "?",
            "subgoal": subgoal.name, "turns_taken": ws.turn_count,
            "deliverable_chars": len(ws.deliverable or ""),
        })
        # ── Tell frontend this workspace is done ──────────────────────────
        self._emit_event({
            "type": "workspace_done",
            "timestamp": time.time(),
            "agent_id": agent.id,
            "model": agent.model,
            "role": member.role.name if member else agent.id,
            "subgoal": subgoal.name,
            "phase_type": subgoal.phase_type,
            "turns_taken": ws.turn_count,
            "rejection_count": ws.rejection_count,
            "status": ws.status,
            "deliverable_preview": (ws.deliverable or "")[:300],
        })
        return ws

    def _run_team_agents(self, agent_members: List, subgoal, upstream: Dict[int, str],
                         user_prompt: str) -> AgentWorkspace:
        """
        Run multiple agents in a short collaborative mini-loop on one subgoal
        (used for convergence nodes, e.g. Combined Analysis).
        Ends with a synthesis turn where the primary agent consolidates output.
        """
        primary_agent, primary_member = agent_members[0]
        ws = AgentWorkspace(agent=primary_agent, member=primary_member,
                            subgoal=subgoal, upstream_deliverables=dict(upstream))
        ws.status = "running"
        max_rounds = max(3, subgoal.estimated_turns)
        turn = 0

        with self._workspace_lock:
            self._active_workspace_count += 1

        self.logger.log_system(
            f"[TEAM] Starting: {subgoal.name} "
            f"({len(agent_members)} agents: {', '.join(m.role.name for _, m in agent_members if m)})"
        )
        self._emit_event({
            "type": "team_task_start", "timestamp": time.time(),
            "agents": [a.id for a, _ in agent_members],
            "roles": [m.role.name for _, m in agent_members if m],
            "subgoal": subgoal.name,
        })

        for _ in range(max_rounds):
            for agent, member in agent_members:
                ctx = self._build_team_context(agent, member, ws, user_prompt)
                try:
                    response = chat_completion(
                        system_prompt=ctx["system_prompt"],
                        user_prompt=ctx["user_prompt"],
                        model=self._select_cost_aware_model(agent.model, subgoal.phase_type),
                        base_url=self.base_url,
                        timeout=120.0,
                        max_tokens=self._get_max_tokens_for_phase(agent.model),
                        temperature=0.7,
                    )
                except Exception as e:
                    err = str(e)
                    is_network_error = any(x in err for x in ("HTTP", "timed out", "Connection", "Request failed"))
                    if not is_network_error:
                        raise
                    print(f"  [Team] {member.role.name if member else agent.id}: {e}")
                    continue
                turn += 1
                self._process_workspace_response(agent, member, response, ws, turn)
            ws.turn_count = turn
            if self._agent_task_complete(ws):
                break

        # Synthesis pass: primary agent consolidates team output
        ws.deliverable = self._synthesize_team_deliverable(
            primary_agent, primary_member, ws, user_prompt
        )
        ws.deliverable = self._deliberate_before_handoff(ws, user_prompt)
        ws.status = "complete"

        with self._workspace_lock:
            self._active_workspace_count = max(0, self._active_workspace_count - 1)

        return ws

    def _synthesize_team_deliverable(self, agent, member, ws: AgentWorkspace,
                                     user_prompt: str) -> str:
        """Ask the primary agent to consolidate team discussion into one clean deliverable."""
        if not ws.private_messages:
            return f"[{ws.subgoal.name}: No team output produced]"

        discussion = "\n\n".join(
            f"**{m.agent_id} ({self.agent_roles.get(m.agent_id, '?')})**: {m.content}"
            for m in ws.private_messages
        )
        try:
            resp = chat_completion(
                system_prompt=(
                    f"You are {member.role.name if member else 'the synthesizer'}. "
                    "Synthesize team discussion into one clean deliverable."
                ),
                user_prompt=(
                    f"Task: {user_prompt}\n\nTeam discussion:\n{discussion[:3000]}\n\n"
                    f"Synthesize all perspectives into ONE clean, non-redundant deliverable "
                    f"for: {ws.subgoal.description}\n"
                    "Remove repetition. Keep unique insights. Use clear headings."
                ),
                model=agent.model,
                base_url=self.base_url,
                timeout=120.0,
                max_tokens=1200,
                temperature=0.3,
            )
            return resp.text.strip()
        except Exception:
            return self._extract_deliverable(ws)

    def _get_adaptive_turn_budget(self, subgoal, role_name: str) -> int:
        """
        Compute turn budget from cross-session handoff memory.
        Falls back to conservative defaults if no history exists.
        """
        defaults = {"research": 10, "analysis": 8, "draft": 14, "review": 8}
        base = max(defaults.get(subgoal.phase_type, 10), subgoal.estimated_turns * 2)

        if self.memory:
            hm = self.memory.get_handoff_memory()
            key = f"{subgoal.phase_type}_{role_name.lower().replace(' ', '_')[:20]}"
            hist = hm.get(key, {})
            if hist.get("runs", 0) >= 3:
                avg_hus = hist.get("avg_hus", 0.5)
                avg_turns = hist.get("avg_turns", base)
                if avg_hus >= 0.72:
                    return max(6, int(avg_turns * 1.1))
                elif avg_hus < 0.45:
                    return min(base * 2, int(avg_turns * 1.3))

        return base

    def _score_handoff_quality(self, upstream_ws: AgentWorkspace,
                                downstream_ws: AgentWorkspace) -> float:
        """
        Handoff Utility Score (HUS): measures how useful an upstream deliverable
        was to the downstream agent.  High early TIS in the downstream workspace
        indicates the input was clear and actionable.
        """
        early = downstream_ws.private_messages[:min(3, len(downstream_ws.private_messages))]
        if not early:
            return 0.5

        avg_tis = sum(m.metrics.get("TIS", 0.5) for m in early) / len(early)

        # Penalise if downstream agent asked many clarifying questions (unclear input)
        early_content = " ".join(m.content for m in early)
        question_penalty = min(0.20, early_content.count("?") * 0.04)

        return max(0.0, min(1.0, avg_tis - question_penalty))

    def _check_bounce_back(self, ws: AgentWorkspace):
        """
        Detect if a downstream workspace is struggling with its upstream input
        and request a revision from the upstream agent.
        Returns (upstream_subgoal_id, feedback_str) or None.
        """
        early = ws.private_messages[:min(3, len(ws.private_messages))]
        if len(early) < 2 or not ws.upstream_deliverables:
            return None

        early_content = " ".join(m.content for m in early)
        avg_tis = sum(m.metrics.get("TIS", 0.5) for m in early) / len(early)
        el = early_content.lower()

        signals = [
            avg_tis < 0.35,
            el.count("?") >= 5,
            "unclear" in el,
            ("missing" in el and "?" in early_content),
            "insufficient" in el,
            "not enough" in el,
            "need more" in el,
        ]
        if sum(signals) < 2:
            return None

        deps = getattr(ws.subgoal, "dependencies", [])
        if not deps:
            return None

        target_sid = deps[-1]
        feedback = (
            f"Your deliverable was returned for revision by {ws.member.role.name if ws.member else 'downstream agent'}.\n"
            f"Issues detected:\n{early_content[:500]}\n\n"
            "Please revise and expand your deliverable to address these gaps. Be more specific and complete."
        )
        return target_sid, feedback

    def _maybe_spawn_emergent_subgoal(self, ws: AgentWorkspace,
                                       assignments: Dict[int, List],
                                       agents: List) -> Optional[Subgoal]:
        """
        Inspect a finished workspace deliverable for signals that a new
        specialist subgoal is needed (e.g., a legal or security review).
        If found and not already covered, appends a new DAG node.
        """
        if not ws.deliverable or not self.phase_controller:
            return None

        dl = ws.deliverable.lower()
        emergent_patterns = {
            "legal":    ["legal review", "legal implications", "compliance check", "regulatory"],
            "security": ["security review", "vulnerability", "security audit", "threat assessment"],
            "financial": ["cost analysis", "financial impact", "budget review", "roi analysis"],
        }

        existing_names = [sg.name.lower() for sg in self.phase_controller.subgoals]

        for category, signals in emergent_patterns.items():
            if any(sig in dl for sig in signals):
                if not any(category in e for e in existing_names):
                    new_id = max(sg.id for sg in self.phase_controller.subgoals) + 1
                    new_sg = Subgoal(
                        id=new_id,
                        name=f"{category.title()} Review",
                        description=f"Conduct {category} review based on needs identified during execution",
                        completion_criteria=f"{category.title()} concerns identified and addressed",
                        estimated_turns=4,
                        phase_type="review",
                        dependencies=[ws.subgoal.id],
                    )
                    print(f"[EMERGENT] Spawning: {new_sg.name} (triggered by {ws.subgoal.name})")
                    self._emit_event({
                        "type": "emergent_subgoal", "timestamp": time.time(),
                        "subgoal": new_sg.name, "triggered_by": ws.subgoal.name,
                        "category": category,
                    })
                    # Assign to an available agent
                    avail = [
                        (next((a for a in agents if a.id == m.agent_id), None), m)
                        for m in (self.team_blueprint.members if self.team_blueprint else [])
                    ]
                    avail = [(a, m) for a, m in avail if a]
                    assignments[new_id] = [avail[-1]] if avail else [(agents[0], None)]
                    return new_sg
        return None

    def _run_fallback(self, user_prompt: str, agents: List) -> List[AcceptedMessage]:
        """Minimal single-agent fallback used when Tier-2 is unavailable."""
        print("[Fallback] Running single-agent mode (Tier-2 unavailable).")
        agent = agents[0]
        required_outputs = self._extract_explicit_output_paths(user_prompt)
        required_output_paths = [p for p in [required_outputs.get("markdown"), required_outputs.get("pdf")] if p]
        verified_required_outputs: Dict[str, bool] = {p: False for p in required_output_paths}

        # Register agent with the dashboard so Active Agents panel populates
        if hasattr(self.display, 'workspace_start'):
            self.display.workspace_start(agent.id, "general", user_prompt[:50])

        # Clamp system_prompt so we never blow Groq's context window
        sys_prompt = agent.system_prompt[:1500] if len(agent.system_prompt) > 1500 else agent.system_prompt
        tool_successes  = 0
        tool_failures = 0
        verify_success = False
        text_completions = 0
        requires_verify_gate = all(
            k in (user_prompt or "").lower()
            for k in ("verify", "non-empty")
        )
        if required_output_paths:
            requires_verify_gate = True

        for turn in range(1, min(self.max_turns + 1, 12)):  # cap fallback at 12 turns
            try:
                output_contract = ""
                if required_outputs.get("markdown") or required_outputs.get("pdf"):
                    output_contract = (
                        "Required output contract (MUST follow exactly):\n"
                        f"- Write markdown to: {required_outputs.get('markdown') or '(none specified)'}\n"
                        f"- Generate PDF to: {required_outputs.get('pdf') or '(none specified)'}\n"
                        "- Use write_text_file for markdown and convert_markdown_to_pdf for PDF.\n"
                        "- Then call verify_file on each required output with min_size_bytes >= 1.\n"
                    )
                tools = self.tool_registry.get_schemas(
                    model=agent.model,
                    task_hint=f"{user_prompt}\n{output_contract}",
                    phase_type="general",
                ) if hasattr(self, 'tool_registry') and self.tool_registry else None
                resp = chat_completion(
                    system_prompt=sys_prompt,
                    user_prompt=(
                        f"Task: {user_prompt}\n\n"
                        f"Turn {turn}: Use tools to complete this task.\n"
                        f"Environment: Windows shell.\n"
                        f"Use Windows-compatible commands/tools (PowerShell/cmd semantics), avoid unix-only commands.\n"
                        f"Before final summary, verify outputs explicitly (file exists, non-empty, expected content).\n"
                        f"{output_contract}\n"
                        f"When done, summarise what you did and include verification results."
                    ),
                    model=agent.model,
                    base_url=self.base_url,
                    timeout=120.0,
                    max_tokens=800,   # keep response short to save context
                    temperature=0.3,
                    tools=tools,
                    tool_choice="auto",
                )

                # Handle tool calls
                if resp.tool_calls:
                    for tc in resp.tool_calls:
                        fn_name = tc.get("function", {}).get("name", "")
                        fn_args_raw = tc.get("function", {}).get("arguments", "{}")
                        try:
                            import json as _json
                            fn_args = _json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                        except Exception:
                            fn_args = {}

                        if hasattr(self, 'display') and hasattr(self.display, 'tool_execution'):
                            self.display.tool_execution(agent.id, fn_name)

                        try:
                            tool_result = self.tool_registry.execute(fn_name, fn_args, self.permission_enforcer)
                            if any(x in tool_result for x in ("EXIT_CODE: 0", "WRITE_OK", "ok=True", "ok=true")):
                                tool_successes += 1
                            if any(x in tool_result for x in ("EXIT_CODE: 1", "EXIT_CODE: 2", "Error:", "STDERR:", "ok=False", "ok=false")):
                                tool_failures += 1
                            if fn_name == "verify_file" and ("ok=True" in tool_result or "ok=true" in tool_result):
                                verify_success = True
                                verified_target = str(fn_args.get("path") or fn_args.get("file_path") or "").strip()
                                for p in list(verified_required_outputs.keys()):
                                    if p.lower() == verified_target.lower():
                                        verified_required_outputs[p] = True
                            if fn_name == "write_text_file" and required_outputs.get("markdown"):
                                md_path = str(required_outputs.get("markdown") or "")
                                written_path = str(fn_args.get("path") or "").strip()
                                if written_path.lower() == md_path.lower() and any(x in tool_result for x in ("WRITE_OK", "ok=True", "ok=true")):
                                    self._verify_required_output_path(md_path, verified_required_outputs)
                            if fn_name == "convert_markdown_to_pdf" and required_outputs.get("pdf"):
                                pdf_path = str(required_outputs.get("pdf") or "")
                                out_path = str(fn_args.get("output_path") or "").strip()
                                if out_path.lower() == pdf_path.lower() and any(x in tool_result for x in ("PDF_OK", "ok=True", "ok=true")):
                                    self._verify_required_output_path(pdf_path, verified_required_outputs)
                        except Exception as te:
                            tool_result = f"Tool error: {te}"
                            tool_failures += 1

                        self.accepted.append(AcceptedMessage(
                            turn=turn, agent_id=agent.id, model=agent.model,
                            content=f"[TOOL] {fn_name}({fn_args}) -> {tool_result}",
                            metrics={"TIS": 0.9},
                            timestamp=time.time(), phase="general", phase_id=0,
                        ))

                    # Update dashboard progress
                    if hasattr(self.display, 'turn_accepted'):
                        self.display.turn_accepted(agent.id, "general", turn, f"Tool:{fn_name}", 0.9)

                    # If we've run several tools successfully, ask for a summary next turn
                    if tool_successes >= 4:
                        if required_output_paths:
                            missing = [p for p, ok in verified_required_outputs.items() if not ok]
                            if missing:
                                continue
                        # Force a final summary turn
                        try:
                            summary_resp = chat_completion(
                                system_prompt=sys_prompt,
                                user_prompt=f"Task complete. Summarise what you did for: {user_prompt}",
                                model=agent.model,
                                base_url=self.base_url,
                                timeout=60.0,
                                max_tokens=400,
                                temperature=0.3,
                            )
                            if summary_resp.text:
                                self.accepted.append(AcceptedMessage(
                                    turn=turn+1, agent_id=agent.id, model=agent.model,
                                    content=self._sanitize_message(summary_resp.text),
                                    metrics={"TIS": 0.8},
                                    timestamp=time.time(), phase="general", phase_id=0,
                                ))
                        except Exception:
                            pass
                        break  # done — don't loop forever after task success
                    continue

                if resp.text:
                    cleaned = self._sanitize_message(resp.text)
                    self.accepted.append(AcceptedMessage(
                        turn=turn, agent_id=agent.id, model=agent.model,
                        content=cleaned, metrics={"TIS": 0.7},
                        timestamp=time.time(), phase="general", phase_id=0,
                    ))
                    if hasattr(self.display, 'turn_accepted'):
                        self.display.turn_accepted(agent.id, "general", turn, cleaned[:50], 0.7)
                    text_completions += 1
                    # Stop after first clean text summary (no more tools needed)
                    if requires_verify_gate and not verify_success:
                        # Do not allow completion claims without explicit verify_file success.
                        continue
                    if required_output_paths and not all(verified_required_outputs.values()):
                        continue
                    if text_completions >= 1 and tool_successes >= 1 and tool_failures == 0:
                        break
                    if text_completions >= 2:
                        break

            except Exception as e:
                print(f"[Fallback] Turn {turn} error: {e}")
                break


        # Mark agent as done in the dashboard
        if hasattr(self.display, 'workspace_done'):
            self.display.workspace_done(agent.id, len(self.accepted), time.time() - self._run_start_time)

        if required_output_paths and not all(verified_required_outputs.values()):
            self.termination_reason = "failed_verification_gate"
        elif requires_verify_gate and not verify_success:
            self.termination_reason = "failed_verification_gate"
        elif not any(
            (msg.content or "").strip() and not (msg.content or "").lstrip().startswith("[TOOL]")
            for msg in self.accepted
        ):
            self.termination_reason = self.termination_reason or "tool_only_run"
            return self.accepted
        self._finalize_conversation()
        return self.accepted

    def _extract_explicit_output_paths(self, user_prompt: str) -> Dict[str, str]:
        """Extract explicit markdown/pdf output paths from user prompt text."""
        text = str(user_prompt or "")
        # Capture Windows absolute paths with .md/.pdf suffix, tolerating spaces.
        candidates = re.findall(r'[A-Za-z]:\\[^\n\r]*?\.(?:md|pdf)', text, flags=re.IGNORECASE)
        normalized: List[str] = []
        for c in candidates:
            cleaned = str(c).strip().strip('"').strip("'").rstrip(".,;:")
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        md_path = next((p for p in normalized if p.lower().endswith(".md")), "")
        pdf_path = next((p for p in normalized if p.lower().endswith(".pdf")), "")
        return {"markdown": md_path, "pdf": pdf_path}

    def _verify_required_output_path(self, path: str, verified_map: Dict[str, bool]) -> None:
        """Run deterministic verify_file for a required artifact path."""
        if not path or path not in verified_map:
            return
        try:
            result = self.tool_registry.execute(
                "verify_file",
                {"path": path, "min_size_bytes": 1},
                self.permission_enforcer,
            )
            if "ok=True" in result or "ok=true" in result:
                verified_map[path] = True
        except Exception:
            return

    def _normalize_planning_prompt(self, prompt: str) -> str:
        """Normalize obvious typos/noise for internal planning only."""
        if not prompt:
            return ""
        s = prompt.strip()
        s = re.sub(r'^\s*roduce\b', 'produce', s, flags=re.IGNORECASE)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _phase_word_budget(self, phase_type: str, role_name: str = "") -> int:
        base = {"research": 420, "analysis": 520, "draft": 900, "review": 520, "general": 500}.get(phase_type or "general", 500)
        rl = (role_name or "").lower()
        if any(k in rl for k in ["review", "auditor", "security"]):
            base = int(base * 0.9)
        return base

    def _emit_event(self, event: Dict[str, Any]) -> None:
        """Emit event to external callback if configured."""
        if not self.event_callback:
            return
        try:
            self.event_callback(event)
        except Exception:
            # Avoid crashing on callback errors
            pass
    
    def _initialize_tier1(self, user_prompt: str, agents: List, subgoals: List = None) -> None:
        """Initialize Tier-1 team formation (role bidding based on DAG subgoals)"""
        try:
            msg = "Initiating Tier-1: DAG Subgoal Bidding..."
            if hasattr(self, 'display') and hasattr(self.display, 'bus_event'):
                self.display.bus_event("SYSTEM", "DAG", msg)
            else:
                self.logger.log_system(msg)
            
            # Form team using TeamFormation module
            team_former = TeamFormation(
                base_url=self.base_url,
                agents=agents,
                event_callback=self._emit_event,
                memory=self.memory,
                task_analysis=self.task_analysis,
                is_renegotiation=False,
                subgoals=subgoals,
            )
            self.team_blueprint = team_former.form_team(user_prompt)
            
            # Apply roles to agent states
            for member in self.team_blueprint.members:
                self.agent_roles[member.agent_id] = member.role.name
            
            # LOGGING: Capture Team Formation Event
            event = {
                "type": "team_formation",
                "timestamp": time.time(),
                "team": [asdict(m) for m in self.team_blueprint.members],
                "formation_turns": self.team_blueprint.formation_turns,
                "negotiation": list(team_former.formation_messages),
                "negotiation_policy": getattr(self.team_blueprint, "negotiation_policy", {}),
                "consensus_strength": getattr(self.team_blueprint, "consensus_strength", 0.0)
            }
            self.full_log.append(event)
            self._emit_event(event)
            
            # UI: Print Beautiful Team Table
            self.logger.print_team([asdict(m) for m in self.team_blueprint.members])

        except Exception as e:
            self.logger.print_error(f"Tier-1 Initialization failed: {e} — using default roles")
            # Minimal fallback: assign static roles, no second negotiation round
            roles = ["Researcher", "Analyst", "Writer", "Reviewer"]
            for i, agent in enumerate(agents):
                role = roles[i % len(roles)]
                self.agent_roles[agent.id] = role
            self.logger.log_system(
                f"Fallback roles assigned: "
                + ", ".join(f"{a.id}={roles[i%len(roles)]}" for i,a in enumerate(agents))
            )

        except Exception as inner_e:
            if self.debug:
                print(f"[Tier-1] Fallback also failed: {inner_e}, continuing without blueprint")
    
    def _gate(self, metrics: Dict[str, float], thresholds: Thresholds, protocol_viol: List[str]) -> Tuple[bool, List[str]]:
        """
        Enhanced AGNN Tier-0 Gate with component-level validation.
        """
        # Extract Tier-0 metrics
        TIS = float(metrics.get("TIS", 0.0))
        SD = float(metrics.get("SD", 0.0))
        RC = float(metrics.get("RC", 0.0))
        IS = float(metrics.get("IS", 0.0))
        EIC = float(metrics.get("EIC", 0.0))
        St = float(metrics.get("St", 0.0))
        
        # Legacy metrics
        tok = float(metrics.get("tok", 0.0))

        reasons: List[str] = []

        # Hard rejects (protocol violations, too long)
        if protocol_viol:
            reasons.extend(protocol_viol)
        if tok > thresholds.T_max_hard:
            reasons.append("Too Long (tok > T_max_hard)")
        if reasons:
            return False, reasons

        # ========== ENHANCED TIER-0 AGNN GATE ==========
        # Enhanced gate with stricter component validation
        
        # Check individual TIS components first (catch specific issues)
        component_failures = []
        
        # Semantic Distance - catch repetitive content (tightened)
        if SD < thresholds.tau_SD:
            component_failures.append(f"Repetitive content (SD={SD:.2f} < {thresholds.tau_SD})")
        
        # Reciprocal Coherence - ensure relevance (tightened)
        if RC < thresholds.tau_RC:
            component_failures.append(f"Low relevance (RC={RC:.2f} < {thresholds.tau_RC})")
        
        # Information Contribution - prevent empty content (tightened)
        if EIC < thresholds.tau_EIC:
            component_failures.append(f"Low information value (EIC={EIC:.2f} < {thresholds.tau_EIC})")
        
        # Interaction Smoothness - maintain flow
        if IS < thresholds.tau_IS:
            component_failures.append(f"Poor conversation flow (IS={IS:.2f} < {thresholds.tau_IS})")
        
        # Stability Score - ensure consistency
        if St < thresholds.tau_St:
            component_failures.append(f"Unstable conversation (St={St:.2f} < {thresholds.tau_St})")
        
        # Reject if any critical component fails (stricter than just TIS)
        if component_failures:
            return False, component_failures
        
        # Final TIS check (tightened threshold)
        if TIS >= thresholds.tau_TIS:
            return True, []
        
        # If we reach here, TIS is below threshold but components passed
        reasons.append(f"Overall TIS below threshold ({TIS:.3f} < {thresholds.tau_TIS})")
        return False, reasons
    
    def _get_max_tokens_for_phase(self, model_id: Optional[str] = None) -> int:
        """
        Determine max_tokens dynamically based on current phase type and model's actual context window.
        """
        # Default token limit (for non-Tier-2 or unknown phases)
        default_tokens = 120
        
        if not self.enable_tier2 or not self.phase_controller:
            return default_tokens
        
        current_phase = self.phase_controller.current_phase
        if not current_phase:
            return default_tokens
            
        # Get true context length from LM Studio, fallback to historical AGNN hardcode if failed
        ctx_length = 4096
        if model_id and hasattr(self, 'model_manager'):
            ctx_length = self.model_manager.get_context_length(model_id, default=4096)

        # Baseline hardcoded fallback minimums in case model manager returns tiny sizes
        fallback_minimums = {
            "research": 1000,
            "analysis": 1500,
            "draft": 2500,
            "review": 1000,
        }
        
        # We assign a percentage of the total context window to output tokens, capped at reasonable sizes so we 
        # don't accidentally ask for 64,000 output tokens on a 128k model (which takes forever).
        # We ensure it's at least the fallback minimum (as long as it doesn't exceed 80% of total config).
        phase_type = current_phase.phase_type
        
        if phase_type == "draft":
            # Drafts need huge output space. We allow up to 6000 output tokens or 50% of context
            raw_target = min(int(ctx_length * 0.5), 6000)
            target = max(fallback_minimums["draft"], raw_target)
        elif phase_type == "analysis":
            raw_target = min(int(ctx_length * 0.35), 4000)
            target = max(fallback_minimums["analysis"], raw_target)
        else: # research, review, general
            raw_target = min(int(ctx_length * 0.25), 2500)
            target = max(fallback_minimums.get(phase_type, 1000), raw_target)
            
        # Hard limit: Output tokens can never exceed 80% of total context length minus 500 (leaving room for prompt)
        absolute_max = int((ctx_length * 0.8) - 500)
        return min(target, max(500, absolute_max))
    
    def _sanitize_message(self, text: str) -> str:
        """Remove speaker labels and protocol violations"""
        # Remove speaker labels
        cleaned = SPEAKER_LABEL_REGEX.sub('', text).strip()
        
        # Remove protocol violations
        if PROTOCOL_RULES.forbid_speaker_labels:
            # Additional cleanup for common patterns
            cleaned = re.sub(r'^(Agent[A-Z]|Assistant|System):\s*', '', cleaned, flags=re.IGNORECASE)

        # Strip accidental raw function-call markup leaked as plain content.
        cleaned = re.sub(r'<function\([^)]*\)\s*\(.*?\)</function>', '', cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned = re.sub(r'^\s*<function.*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
        
        return cleaned
    
    def _has_invalid_agent_reference(self, content: str, current_agent_id: str) -> bool:
        refs = self.invalid_agent_ref_re.findall(content or "")
        if not refs:
            return False
        valid = set(self.agent_roles.keys())
        valid.add(current_agent_id)
        for r in refs:
            if r not in valid:
                return True
        return False

    def _select_cost_aware_model(self, default_model: str, phase_type: str) -> str:
        models = list(dict.fromkeys(self.models or []))
        if not models:
            return default_model

        def cost_score(m: str) -> float:
            ml = m.lower()
            b = re.search(r'(\d+(?:\.\d+)?)b', ml)
            if b:
                try:
                    return float(b.group(1))
                except Exception:
                    pass
            if '1b' in ml or '2b' in ml:
                return 2.0
            if '7b' in ml or '8b' in ml:
                return 8.0
            return 5.0

        sorted_models = sorted(models, key=cost_score)
        if phase_type in ('research',):
            chosen = sorted_models[0]
        elif phase_type in ('analysis', 'review'):
            chosen = sorted_models[min(1, len(sorted_models)-1)]
        else:  # draft / merge-heavy
            chosen = sorted_models[-1]

        return chosen

    def _update_ifcm(self, current_sender: str, previous_sender: Optional[str], tis: float = 0.0, phase_id: int = 0) -> None:
        """
        Update Information Flow Control Matrix (IFCM) with LDCL logic.
        Tracks Edge Utility: How valuable is the transition previous_sender -> current_sender?
        Update Rule: Q_new(s,a) = Q_old(s,a) + alpha * (Reward - Q_old(s,a))
        """
        if not previous_sender:
            return

        phase_type = None
        if self.enable_tier2 and self.phase_controller:
            phase = self.phase_controller.subgoal_map.get(phase_id)
            if phase:
                phase_type = phase.phase_type

        # Simple IFCM update (direct, no LDCL overhead)
        if previous_sender not in self.ifcm:
            self.ifcm[previous_sender] = {}
        old_utility = self.ifcm[previous_sender].get(current_sender, 0.5)
        alpha = 0.2
        new_utility = old_utility + alpha * (tis - old_utility)
        self.ifcm[previous_sender][current_sender] = new_utility

    def _track_agent_performance(self, agent_id: str, metrics: Dict[str, float], turn: int, phase_id: int = 0) -> None:
        """
        Track agent performance for monitoring and analysis.
        NEW: Comprehensive performance tracking system.
        """
        if agent_id not in self.agent_turn_history:
            self.agent_turn_history[agent_id] = []
        
        # Determine Phase Name
        phase_name = "unknown"
        if self.enable_tier2 and self.phase_controller:
            phase = self.phase_controller.subgoal_map.get(phase_id)
            if phase:
                phase_name = phase.name
                
        # Record turn data
        turn_data = {
            "turn": turn,
            "tis": metrics.get("TIS", 0.0),
            "eic": metrics.get("EIC", 0.0),
            "sd": metrics.get("SD", 0.0),
            "rc": metrics.get("RC", 0.0),
            "actionability": metrics.get("A", 0.0),
            "timestamp": time.time(),
            "phase": phase_name
        }
        
        self.agent_turn_history[agent_id].append(turn_data)
        
        # Calculate rolling averages (last 5 turns)
        recent_turns = self.agent_turn_history[agent_id][-5:]
        avg_tis = sum(t["tis"] for t in recent_turns) / len(recent_turns)
        avg_eic = sum(t["eic"] for t in recent_turns) / len(recent_turns)
        
        # Track phase-specific performance
        if self.enable_tier2 and self.phase_controller and self.phase_controller.current_phase:
            phase_id = str(self.phase_controller.current_phase.id)
            if phase_id not in self.phase_performance_tracking:
                self.phase_performance_tracking[phase_id] = {
                    "phase_name": self.phase_controller.current_phase.name,
                    "agents": {}
                }
            
            if agent_id not in self.phase_performance_tracking[phase_id]["agents"]:
                self.phase_performance_tracking[phase_id]["agents"][agent_id] = {
                    "turns": 0,
                    "total_tis": 0.0,
                    "total_eic": 0.0,
                    "best_tis": 0.0,
                    "worst_tis": 1.0
                }
            
            agent_phase_data = self.phase_performance_tracking[phase_id]["agents"][agent_id]
            agent_phase_data["turns"] += 1
            agent_phase_data["total_tis"] += metrics.get("TIS", 0.0)
            agent_phase_data["total_eic"] += metrics.get("EIC", 0.0)
            agent_phase_data["best_tis"] = max(agent_phase_data["best_tis"], metrics.get("TIS", 0.0))
            agent_phase_data["worst_tis"] = min(agent_phase_data["worst_tis"], metrics.get("TIS", 0.0))
            agent_phase_data["avg_tis"] = agent_phase_data["total_tis"] / agent_phase_data["turns"]
            agent_phase_data["avg_eic"] = agent_phase_data["total_eic"] / agent_phase_data["turns"]
        
        # Debug output for performance tracking
        if self.debug and len(recent_turns) >= 3:
            print(f"[PERF] {agent_id}: Avg TIS={avg_tis:.3f}, Avg EIC={avg_eic:.3f} (last 5 turns)")
    
    def get_agent_performance_summary(self) -> Dict[str, Dict]:
        """Get comprehensive performance summary for all agents"""
        summary = {}
        
        for agent_id, turn_history in self.agent_turn_history.items():
            if not turn_history:
                continue
            
            total_turns = len(turn_history)
            avg_tis = sum(t["tis"] for t in turn_history) / total_turns
            avg_eic = sum(t["eic"] for t in turn_history) / total_turns
            best_tis = max(t["tis"] for t in turn_history)
            worst_tis = min(t["tis"] for t in turn_history)
            
            # Calculate trend (improving/declining)
            if total_turns >= 4:
                first_half = turn_history[:total_turns//2]
                second_half = turn_history[total_turns//2:]
                first_avg = sum(t["tis"] for t in first_half) / len(first_half)
                second_avg = sum(t["tis"] for t in second_half) / len(second_half)
                trend = "improving" if second_avg > first_avg + 0.05 else "declining" if second_avg < first_avg - 0.05 else "stable"
            else:
                trend = "insufficient_data"
            
            summary[agent_id] = {
                "total_turns": total_turns,
                "avg_tis": avg_tis,
                "avg_eic": avg_eic,
                "best_tis": best_tis,
                "worst_tis": worst_tis,
                "trend": trend,
                "current_role": self.agent_roles.get(agent_id, "unknown"),
                "participation_rate": total_turns / len(self.accepted) if self.accepted else 0.0
            }
        
        return summary
    
    def _run_auto_score_calibration(
        self,
        final_document: str,
        user_prompt: str,
        max_chars: int = 3000,
    ) -> dict:
        """
        Auto-Score Calibration (medium priority item 9).

        A separate LLM evaluator reads the final synthesized document and
        scores it across 6 dimensions on a 0-10 scale.  Scores are stored
        in the run log and displayed via self.display.score_display().

        Returns a dict like {
            "coverage": 8.2, "depth": 7.5, "coherence": 9.0,
            "accuracy": 7.0, "actionability": 8.5, "formatting": 9.0,
            "overall": 8.2
        }
        Returns {} if calibration cannot run (no models available).
        """
        if not final_document or not self.models:
            return {}

        # Choose the most reliable available model for evaluation
        evaluator_model = self.models[0]

        doc_excerpt = final_document[:max_chars]
        if len(final_document) > max_chars:
            doc_excerpt += f"\n... [truncated — {len(final_document)} total chars]"

        system_prompt = (
            "You are a strict document quality evaluator. "
            "Evaluate the provided document on 6 dimensions. "
            "Return ONLY a JSON object — no markdown, no explanation.\n\n"
            "Dimensions and rubric:\n"
            "- coverage     (0-10): Does the document address ALL sections of the original task?\n"
            "- depth        (0-10): Are explanations specific, with data, numbers, and examples?\n"
            "- coherence    (0-10): Is the document logically structured and internally consistent?\n"
            "- accuracy     (0-10): Are all stated facts plausible and internally consistent?\n"
            "- actionability(0-10): Are recommendations concrete and easy to act on?\n"
            "- formatting   (0-10): Is the document well formatted with clear headings and structure?\n\n"
            "Return exactly: "
            "{\"coverage\":X,\"depth\":X,\"coherence\":X,\"accuracy\":X,\"actionability\":X,\"formatting\":X}"
        )
        user_prompt_eval = (
            f"Original task: {user_prompt[:400]}\n\n"
            f"Document to evaluate:\n{doc_excerpt}\n\n"
            "Return the JSON scores now."
        )

        try:
            from .llm_client import chat_completion as _cc
            resp = _cc(
                system_prompt=system_prompt,
                user_prompt=user_prompt_eval,
                model=evaluator_model,
                base_url=self.base_url,
                timeout=30.0,
                max_tokens=180,
                temperature=0.0,
            )

            import re as _re
            import json as _json
            m = _re.search(r'\{[^}]+\}', resp.text, _re.DOTALL)
            if not m:
                return {}
            scores = _json.loads(m.group(0))
            # Clamp all scores to [0, 10]
            clean: dict = {}
            for k, v in scores.items():
                try:
                    clean[k] = max(0.0, min(10.0, float(v)))
                except Exception:
                    pass
            if clean:
                overall = sum(clean.values()) / len(clean)
                clean["overall"] = round(overall, 2)
                print("\n[Auto-Score] Document quality calibration:")
                self.display.score_display(clean)
            return clean

        except Exception as e:
            print(f"[Auto-Score] Calibration skipped: {e}")
            return {}

    def _finalize_conversation(self) -> None:
        """Finalize conversation and save comprehensive log"""
        # Calculate final IFCM influence scores for all agents
        ifcm_summary = {}
        filtered_ifcm = None
        if self.ifcm:
            active_agents = set(self.agent_roles.keys()) if self.agent_roles else set()

            # Build filtered IFCM for active agents only
            filtered_ifcm = {}
            for sender in self.ifcm:
                if sender not in active_agents:
                    continue
                targets = {
                    target: value
                    for target, value in self.ifcm[sender].items()
                    if target in active_agents
                }
                filtered_ifcm[sender] = targets

            # Calculate influence scores using filtered IFCM
            def _influence_score(agent_id: str) -> float:
                total = 0.0
                for sender in filtered_ifcm:
                    if agent_id in filtered_ifcm[sender]:
                        total += filtered_ifcm[sender][agent_id]
                return total

            for agent_id in active_agents:
                ifcm_summary[agent_id] = {
                    "influence_score": _influence_score(agent_id),
                    "influences": filtered_ifcm.get(agent_id, {})
                }
        
        # Create comprehensive log entry with all messages
        accepted_messages_data = [
            {
                "turn": msg.turn,
                "agent_id": msg.agent_id,
                "model": msg.model,
                "content": msg.content,
                "metrics": msg.metrics,
                "timestamp": msg.timestamp,
                "phase": msg.phase
            }
            for msg in self.accepted
        ]
        
        
        # Build comprehensive subgoal information
        subgoals_data = []
        if self.phase_controller:
            completed_ids = set(getattr(self.phase_controller, "completed_phase_ids", set()))
            active_ids = set(getattr(self.phase_controller, "active_phase_ids", set()))
            for subgoal in self.phase_controller.subgoals:
                assigned_agent = self.phase_controller.get_assigned_agent(subgoal.id)
                if subgoal.id in completed_ids:
                    status = "completed"
                elif subgoal.id in active_ids:
                    status = "in_progress"
                else:
                    status = "pending"
                subgoal_info = {
                    "id": subgoal.id,
                    "name": subgoal.name,
                    "description": subgoal.description,
                    "phase_type": subgoal.phase_type,
                    "completion_criteria": subgoal.completion_criteria,
                    "estimated_turns": subgoal.estimated_turns,
                    "assigned_agent": assigned_agent,
                    "status": status
                }
                subgoals_data.append(subgoal_info)
        
        # Build team formation transcript
        team_formation_transcript = None
        if self.team_blueprint:
            team_formation_transcript = {
                "formation_turns": self.team_blueprint.formation_turns,
                "proposed_subgoals": [
                    {
                        "id": sg.id,
                        "name": sg.name,
                        "description": sg.description,
                        "phase_type": sg.phase_type
                    }
                    for sg in self.team_blueprint.subgoals
                ],
                "team_members": [
                    {
                        "agent_id": m.agent_id,
                        "model": m.model,
                        "role": m.role.name,
                        "role_description": m.role.description,
                        "responsibilities": m.role.responsibilities,
                        "confidence": m.confidence
                    }
                    for m in self.team_blueprint.members
                ]
            }
            if self.team_plan:
                team_formation_transcript["team_plan"] = self.team_plan
        
        max_turn = 0
        for msg in self.accepted:
            try:
                max_turn = max(max_turn, int(msg.turn))
            except Exception:
                pass
        for rej in self.rejected_messages:
            try:
                max_turn = max(max_turn, int(rej.get("turn", 0)))
            except Exception:
                pass

        phase_summary = self.phase_controller.get_summary() if self.phase_controller else None
        completed_phases = int(phase_summary.get("completed_phases", 0)) if isinstance(phase_summary, dict) else 0
        total_phases = int(phase_summary.get("total_phases", len(subgoals_data))) if isinstance(phase_summary, dict) else len(subgoals_data)

        log_data = {
            "user_prompt": self.user_prompt,
            "timestamp": datetime.now().isoformat(),
            "termination_reason": self.termination_reason or ("completed" if (self.phase_controller and self.phase_controller.is_complete) else "partial"),
            
            # Summary statistics
            "summary": {
                "accepted_count": len(self.accepted),
                "rejected_count": len(self.rejected_messages),
                "rewrites_count": self.rewrite_count,
                "total_turns": max_turn,
                "total_phases": total_phases,
                "completed_phases": completed_phases
            },
            "kpis": {},  # Removed _compute_run_kpis() temporarily
            
            # Message logs
            "messages": {
                "accepted": accepted_messages_data,
                "rejected": self.rejected_messages,
                "full_transcript": self.full_log  # Complete log with all attempts
            },
            
            # Tier-1: Team Formation
            "tier1_team_formation": team_formation_transcript,
            
            # Tier-2: Subgoals and Phases
            "tier2_execution": {
                "subgoals": subgoals_data,
                "phase_history": self.phase_controller.phase_history if self.phase_controller else [],
                "phase_summary": phase_summary,
                "agent_performance": self.phase_controller.agent_performance if self.phase_controller else {}
            },
            
            # Agent metrics
            "agent_metrics": {
                "ifcm_summary": ifcm_summary,
                "ifcm_matrix": filtered_ifcm if filtered_ifcm is not None else self.ifcm,
                "performance_summary": self.get_agent_performance_summary(),
                "turn_history": self.agent_turn_history,
                "phase_performance": self.phase_performance_tracking
            },
            
            # Granular Events (NEW: Full Event Log)
            "events": [e for e in self.full_log if e.get("type")]
        }
        
        # Save to logs directory
        os.makedirs("agnn/logs", exist_ok=True)
        
        # PERSIST MEMORY (The Brain)
        if self.memory:
            try:
                self.memory.update_ifcm(self.ifcm)
                self.memory.update_session_stats()
                print("[Memory] Brain updated with new session data.")
            except Exception as e:
                print(f"[Memory] Failed to update brain: {e}")
        
        # Create filename from user prompt (sanitized)

        if self.user_prompt:
            # Sanitize user prompt for filename
            prompt_lower = self.user_prompt.lower().strip()
            # Remove special characters, keep only alphanumeric and spaces
            prompt_clean = re.sub(r'[^a-z0-9\s-]', '', prompt_lower)
            # Replace spaces and multiple dashes with single dash
            prompt_clean = re.sub(r'[\s-]+', '-', prompt_clean)
            # Take first 50 characters max
            words = prompt_clean.split('-')[:10]  # First 10 words
            slug = "-".join(words) if words else "conversation"
            # Remove leading/trailing dashes
            slug = slug.strip('-')
            # Ensure it's not empty
            if not slug or len(slug) < 3:
                slug = "conversation"
        else:
            slug = "conversation"
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"agnn/logs/{slug}-{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n[LOG] Conversation saved to: {filename}")

            # --- POST-RUN SYNTHESIS: produce one clean Markdown deliverable ---
            try:
                best_model = self.models[0] if self.models else ""
                synth_result = synthesize(
                    user_prompt=self.user_prompt,
                    accepted_messages=accepted_messages_data,
                    base_url=self.base_url,
                    model=best_model,
                    output_dir="agnn/outputs",
                    slug=slug,
                    handoff_packages=self.handoff_packages if self.handoff_packages else None,
                )
                # ── Tell frontend the document is ready ───────────────────
                doc_text = ""
                if synth_result and isinstance(synth_result, str):
                    doc_text = synth_result
                else:
                    import glob as _g2
                    _mds = sorted(_g2.glob(f"agnn/outputs/{slug}*.md"),
                                  key=lambda p: len(open(p, 'r', encoding='utf-8', errors='replace').read()),
                                  reverse=True)
                    if _mds:
                        doc_text = open(_mds[0], 'r', encoding='utf-8', errors='replace').read()
                if doc_text:
                    self._emit_event({
                        "type": "synthesis_complete",
                        "timestamp": time.time(),
                        "document": doc_text,
                        "chars": len(doc_text),
                    })
                    if "degraded_synthesis" in doc_text.lower() or "synthesis degraded" in doc_text.lower():
                        log_data["synthesis_status"] = "degraded"
                    else:
                        log_data["synthesis_status"] = "ok"
            except Exception as syn_err:
                print(f"[Synthesizer] Skipped due to error: {syn_err}")
                log_data["synthesis_status"] = "failed"

            # --- AUTO-SCORE the final deliverable (two-pass) ---
            try:
                # Pass 1: fast heuristic scorer
                best_deliverable = ""
                if self.accepted:
                    best_deliverable = max(
                        (m.content for m in self.accepted),
                        key=lambda c: len(c)
                    )

                # Try to read the actual synthesized .md file for a richer evaluation
                import glob as _glob
                md_files = sorted(
                    _glob.glob(f"agnn/outputs/{slug}*.md"),
                    key=lambda p: len(open(p, 'r', encoding='utf-8', errors='replace').read()),
                    reverse=True,
                )
                if md_files:
                    try:
                        synth_text = open(md_files[0], 'r', encoding='utf-8', errors='replace').read()
                        if len(synth_text) > 200:
                            best_deliverable = synth_text
                    except Exception:
                        pass

                if best_deliverable and log_data.get("synthesis_status") != "degraded":
                    best_model = self.models[0] if self.models else ""
                    # Fast heuristic scorer
                    scores = score_deliverable(
                        deliverable=best_deliverable,
                        user_prompt=self.user_prompt,
                        model=best_model,
                        base_url=self.base_url,
                    )
                    self.display.score_display(scores)
                    log_data["auto_score"] = scores
                    # ── Tell frontend about the scores ────────────────────
                    self._emit_event({
                        "type": "auto_score",
                        "timestamp": time.time(),
                        "scores": scores,
                    })

                    # Pass 2: LLM-based calibration (more expensive but higher quality)
                    calibrated = self._run_auto_score_calibration(
                        final_document=best_deliverable,
                        user_prompt=self.user_prompt,
                    )
                    if calibrated:
                        log_data["calibrated_score"] = calibrated
                        # Prefer the richer LLM scores for the frontend
                        self._emit_event({
                            "type": "auto_score",
                            "timestamp": time.time(),
                            "scores": calibrated,
                        })

            except Exception as score_err:
                print(f"[Scorer] Skipped: {score_err}")

            if self.debug:
                print(f"  - Rejected: {len(self.rejected_messages)} messages")
                print(f"  - Total log entries: {len(self.full_log)}")

            # Rewrite final log after synthesis/scoring updates.
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] Failed to save log: {e}")
            import traceback
            traceback.print_exc()

        # ── Close the live dashboard ──────────────────────────────────────────
        if hasattr(self.display, 'run_complete'):
            wall = time.time() - self._run_start_time
            out  = filename if 'filename' in dir() else ""
            self.display.run_complete(
                wall,
                len(self.accepted),
                self.rejected_count,
                out,
                self.termination_reason or "partial",
            )


    def _validate_models(self) -> Dict[str, bool]:
        """Test each model before run; return per-model health map."""
        print("[Validation] Testing model reliability...")

        health: Dict[str, bool] = {}
        for model in self.models:
            try:
                is_gemini = model.startswith("models/gemini")
                test_response = chat_completion(
                    system_prompt="Return exactly READY.",
                    user_prompt="health_check",
                    model=model,
                    base_url=self.base_url,
                    timeout=30.0,
                    max_tokens=128 if is_gemini else 32,
                    temperature=0.0
                )
                
                resp_text = (test_response.text or "").strip().lower()
                if not resp_text or ("ready" not in resp_text and resp_text.replace(".", "").strip() != "ready"):
                    print(f"[Validation] WARNING: Model {model} gave unexpected response: {test_response.text[:100]}")
                    health[model] = False
                else:
                    print(f"[Validation] ✓ Model {model} working correctly")
                    health[model] = True
                    
            except Exception as e:
                print(f"[Validation] ERROR: Model {model} failed test: {e}")
                health[model] = False

        healthy_count = sum(1 for ok in health.values() if ok)
        print(f"[Validation] Healthy models: {healthy_count}/{len(self.models)}")
        return health

    def _validate_models(self) -> Dict[str, bool]:
        """Test each model before run; return per-model health map."""
        print("[Validation] Testing model reliability...")

        health: Dict[str, bool] = {}
        for model in self.models:
            try:
                probe_prompt = "Call the list_dirs tool with path='.' and do not answer with prose."
                if hasattr(self, "tool_registry") and self.tool_registry:
                    probe_tools = [
                        self.tool_registry.tools["read_file"].get_schema(),
                        self.tool_registry.tools["list_dirs"].get_schema(),
                    ]
                else:
                    probe_tools = None
                test_response = chat_completion(
                    system_prompt="You must use tools when the user explicitly requests one.",
                    user_prompt=probe_prompt,
                    model=model,
                    base_url=self.base_url,
                    timeout=90.0,
                    max_tokens=80,
                    temperature=0.0,
                    tools=probe_tools,
                    tool_choice="auto",
                )

                tool_calls = test_response.tool_calls or []
                used_list_dirs = any(
                    isinstance(tc, dict) and tc.get("function", {}).get("name") == "list_dirs"
                    for tc in tool_calls
                )
                if used_list_dirs:
                    print(f"[Validation] ✓ Model {model} can issue tool calls")
                    health[model] = True
                elif (test_response.text or "").strip():
                    print(f"[Validation] ✓ Model {model} responded without tools")
                    health[model] = True
                else:
                    preview = (test_response.text or "")[:100]
                    print(f"[Validation] WARNING: Model {model} gave unexpected probe response: {preview}")
                    health[model] = False

            except Exception as e:
                print(f"[Validation] ERROR: Model {model} failed test: {e}")
                health[model] = False

        healthy_count = sum(1 for ok in health.values() if ok)
        print(f"[Validation] Healthy models: {healthy_count}/{len(self.models)}")
        return health

