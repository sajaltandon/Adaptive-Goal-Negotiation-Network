import {
  AppPhase, ChartEvent, DecisionType, MetricPoint,
  PhaseMarker, QualityScores, RunMetrics, SubgoalInfo,
  TeamFormationData, TeamPlanData, AgentWorkspaceState, WorkflowPhase,
  SubgoalStatus
} from '../types';

const API_URL = import.meta.env.VITE_AGNN_API_URL || 'http://localhost:8000';

let eventSource: EventSource | null = null;

type UpdateAgentFn = (id: string, updates: Record<string, any>) => void;
type UpdateTeamFormationFn = (
  next: TeamFormationData | ((prev: TeamFormationData | null) => TeamFormationData | null)
) => void;
type UpdateWorkspaceFn = (id: string, updates: Partial<AgentWorkspaceState>) => void;

const buildWorkspace = (agentId: string, model: string, role: string): AgentWorkspaceState => ({
  agent_id: agentId,
  model,
  primary_role: role,
  secondary_roles: [],
  current_subgoal: undefined,
  subgoal_phase: undefined,
  turn_count: 0,
  max_turns: 12,
  tis_score: 0,
  rejection_count: 0,
  last_message: '',
  status: 'pending',
  elapsed_s: 0,
});

export const startSimulation = async (
  selectedModels: string[],   // ← the models the user picked in the UI
  prompt: string,
  baseUrl: string,
  enableTier2: boolean,
  addMessage: (msg: any) => void,
  updateAgent: UpdateAgentFn,
  updateMetrics: (fn: (prev: RunMetrics) => RunMetrics) => void,
  updateChart: (fn: (prev: MetricPoint[]) => MetricPoint[]) => void,
  updatePhaseMarkers: (fn: (prev: PhaseMarker[]) => PhaseMarker[]) => void,
  updateChartEvents: (fn: (prev: ChartEvent[]) => ChartEvent[]) => void,
  updateTeamFormation: UpdateTeamFormationFn,
  updateTeamPlan: (data: TeamPlanData) => void,
  setAppPhase: (phase: AppPhase) => void,
  setWorkflowPhase: (phase: WorkflowPhase) => void,
  updateSubgoals: (fn: (prev: SubgoalInfo[]) => SubgoalInfo[]) => void,
  updateWorkspace: UpdateWorkspaceFn,
  setQualityScores: (scores: QualityScores) => void,
  setSynthesizedDoc: (doc: string) => void,
  addEventLog: (entry: string) => void,
) => {
  if (eventSource) eventSource.close();

  const statsMap = new Map<string, { turns: number; accepted: number; rewrites: number; rejects: number }>();
  const roleMap = new Map<string, string>();
  const workspaceElapsedTimers = new Map<string, number>();
  let currentPhase: WorkflowPhase | null = null;

  // Build the payload that goes to the backend /runs endpoint.
  // - auto_mode: false  → backend MUST use exactly the models we provide
  // - models: selectedModels → the exact list the user confirmed in the UI
  const payload = {
    prompt,
    base_url: baseUrl,
    models: selectedModels,
    enable_tier2: enableTier2,
    auto_mode: false,
    max_agents: Math.max(2, Math.min(5, selectedModels.length || 4)),
  };

  const resp = await fetch(`${API_URL}/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!resp.ok) throw new Error('Failed to start run');
  const { run_id } = await resp.json();
  setAppPhase(AppPhase.RUNNING);

  eventSource = new EventSource(`${API_URL}/runs/${run_id}/stream`);

  eventSource.onmessage = (event) => {
    let data: any;
    try { data = JSON.parse(event.data); } catch { return; }

    const now = formatTimestamp();

    // ── Task Analysis ──────────────────────────────────────────────────────
    if (data.type === 'task_analysis' && data.analysis) {
      const a = data.analysis;
      addEventLog(`[Analysis] Type: ${a.task_type || 'general'} | Complexity: ${a.complexity} | Team: ${a.team_size}`);
      return;
    }

    // ── Model Selection ────────────────────────────────────────────────────
    if (data.type === 'auto_models_selected' && Array.isArray(data.models)) {
      addEventLog(`[Models] Selected ${data.models.length}: ${data.models.join(', ')}`);
      return;
    }

    // ── Agents Initialized ─────────────────────────────────────────────────
    if (data.type === 'agents_initialized' && Array.isArray(data.agents)) {
      data.agents.forEach((agent: any) => {
        const id = String(agent.agent_id || 'AgentX');
        const model = String(agent.model || id);
        statsMap.set(id, { turns: 0, accepted: 0, rewrites: 0, rejects: 0 });
        updateAgent(id, {
          name: model, role: 'Unassigned', status: 'idle',
          isThinking: false, confidence: 0.9,
          provider: 'LM Studio', version: 'Auto', capabilities: ['Autonomous'],
        });
        updateWorkspace(id, buildWorkspace(id, model, 'Unassigned'));
      });
      addEventLog(`[Init] ${data.agents.length} agents initialized`);
      return;
    }

    // ── Tier-1 Roles Generated ─────────────────────────────────────────────
    if (data.type === 'roles_generated' && Array.isArray(data.roles)) {
      updateTeamFormation(prev => ({
        members: prev?.members || [],
        negotiation: prev?.negotiation || [],
        formationTurns: prev?.formationTurns || 0,
        roles: data.roles,
      }));
      addEventLog(`[Tier-1] ${data.roles.length} dynamic roles generated`);
      return;
    }

    // ── Negotiation Turn ───────────────────────────────────────────────────
    if (data.type === 'negotiation_turn') {
      updateTeamFormation(prev => {
        const cur = prev || { members: [], negotiation: [], formationTurns: 0, roles: [] };
        return {
          ...cur,
          negotiation: [...cur.negotiation, {
            turn: data.turn, agent: data.agent, content: data.content,
            proposed_role: data.proposed_role, final_role: data.final_role,
            negotiation_score: data.negotiation_score,
            phase: data.phase,
          }],
          formationTurns: Math.max(cur.formationTurns, Number(data.turn) || 0),
        };
      });
      return;
    }

    // ── Consensus Meter ────────────────────────────────────────────────────
    if (data.type === 'consensus_meter') {
      updateTeamFormation(prev => prev ? { ...prev, consensusStrength: data.strength } : prev);
      return;
    }

    // ── Team Formation ─────────────────────────────────────────────────────
    if (data.type === 'team_formation' && Array.isArray(data.team)) {
      data.team.forEach((member: any) => {
        const role = member.role?.name || member.role || 'Unassigned';
        updateAgent(member.agent_id, { role });
        updateWorkspace(member.agent_id, {
          primary_role: role,
          secondary_roles: (member.secondary_roles || []).map((r: any) => r?.name || r),
        });
      });
      updateTeamFormation(prev => ({
        members: data.team,
        negotiation: prev?.negotiation || [],
        formationTurns: data.formation_turns || prev?.formationTurns || 0,
        consensusStrength: data.consensus_strength ?? prev?.consensusStrength,
        roles: prev?.roles || [],
      }));
      addEventLog(`[Tier-1] Team formed: ${data.team.map((m: any) => m.role?.name || m.role).join(', ')}`);
      return;
    }

    // ── Team Plan ──────────────────────────────────────────────────────────
    if (data.type === 'team_plan' && data.plan) {
      updateTeamPlan(data.plan as TeamPlanData);
      return;
    }

    // ── Subgoal DAG Initialized ────────────────────────────────────────────
    // (Both 'subgoals_initialized' and legacy 'subgoals_created' are handled)
    if ((data.type === 'subgoals_initialized' || data.type === 'subgoals_created') && Array.isArray(data.subgoals)) {
      updateSubgoals(data.subgoals.map((sg: any) => ({
        id: sg.id || sg.subgoal_id,
        name: sg.name,
        description: sg.description || '',
        phase_type: sg.phase_type || 'research',
        dependencies: sg.dependencies || [],
        assigned_agent: sg.assigned_agent,
        assigned_role: sg.assigned_role,
        status: (sg.status || 'pending') as SubgoalStatus,
        turns: 0,
        max_turns: sg.estimated_turns || 8,
      })));
      addEventLog(`[Tier-2] DAG: ${data.subgoals.length} subgoals initialized`);
      return;
    }

    // ── Workspace Started ──────────────────────────────────────────────────
    if (data.type === 'workspace_started') {
      const id = data.agent_id;
      const subgoalName = data.subgoal || data.subgoal_name;
      updateWorkspace(id, {
        model: data.model || undefined,
        primary_role: data.role || undefined,
        current_subgoal: subgoalName,
        subgoal_phase: data.phase_type,
        status: 'generating',
        turn_count: 0,
        max_turns: data.max_turns || 12,
      });
      updateSubgoals(prev => prev.map(sg =>
        sg.name === subgoalName ? { ...sg, status: 'running', assigned_agent: id } : sg
      ));
      // Start elapsed timer
      const start = Date.now();
      const timer = window.setInterval(() => {
        updateWorkspace(id, { elapsed_s: Math.floor((Date.now() - start) / 1000) });
      }, 1000);
      workspaceElapsedTimers.set(id, timer);
      addEventLog(`[${id}] Starting: ${subgoalName}`);
      return;
    }

    // ── Workspace Done ─────────────────────────────────────────────────────
    if (data.type === 'workspace_done') {
      const id = data.agent_id;
      const subgoalName = data.subgoal || data.subgoal_name;
      const succeeded = data.status === 'complete' || data.status === 'completed';
      updateWorkspace(id, {
        status: 'accepted',
        rejection_count: data.rejection_count ?? undefined,
        turn_count: data.turns_taken ?? undefined,
      });
      clearInterval(workspaceElapsedTimers.get(id));
      workspaceElapsedTimers.delete(id);
      updateSubgoals(prev => prev.map(sg =>
        sg.name === subgoalName
          ? { ...sg, status: succeeded ? 'completed' : 'failed' }
          : sg
      ));
      addEventLog(`[${id}] ✓ Done: ${subgoalName} (${data.turns_taken} turns)`);
      return;
    }

    // ── Hot Swap ───────────────────────────────────────────────────────────
    if (data.type === 'hot_swap') {
      const id = data.agent_id;
      const newModel = data.new_model || data.to_model;
      const oldModel = data.old_model || data.from_model;
      updateWorkspace(id, { model: newModel, status: 'hot-swapping' });
      updateAgent(id, { name: newModel });
      addEventLog(`[⚡ Hot-Swap] ${id}: ${oldModel} → ${newModel}`);
      updateChartEvents(prev => [...prev, {
        timestamp: now, label: `${id} hot-swap → ${newModel}`, kind: 'hotswap' as const
      }]);
      // Revert to generating after 2s
      setTimeout(() => updateWorkspace(id, { status: 'generating' }), 2000);
      return;
    }

    // ── Phase Change ───────────────────────────────────────────────────────
    if (data.type === 'phase_change') {
      const phase = toWorkflowPhase(data.phase);
      if (phase) {
        updatePhaseMarkers(prev => [...prev, { timestamp: now, label: phase }]);
        setWorkflowPhase(phase);
        currentPhase = phase;
        addEventLog(`[Phase] → ${phase}`);
      }
      return;
    }

    // ── Message ────────────────────────────────────────────────────────────
    if (data.type === 'message') {
      const decision = toDecision(data.decision);
      const phase = toWorkflowPhase(data.phase) || currentPhase || WorkflowPhase.RESEARCH;
      const metrics = data.metrics || {};
      const agentId = data.agent_id;

      // Track role changes
      const nextRole = data.agent_role || '';
      if (nextRole && roleMap.get(agentId) !== nextRole) {
        roleMap.set(agentId, nextRole);
        updateAgent(agentId, { role: nextRole });
        updateWorkspace(agentId, { primary_role: nextRole });
      }

      // Update workspace live view
      const currentStats = statsMap.get(agentId) || { turns: 0, accepted: 0, rewrites: 0, rejects: 0 };
      const nextStats = {
        turns: currentStats.turns + 1,
        accepted: decision === DecisionType.ACCEPT ? currentStats.accepted + 1 : currentStats.accepted,
        rewrites: decision === DecisionType.REWRITE ? currentStats.rewrites + 1 : currentStats.rewrites,
        rejects: decision === DecisionType.REJECT ? currentStats.rejects + 1 : currentStats.rejects,
      };
      statsMap.set(agentId, nextStats);

      updateWorkspace(agentId, {
        turn_count: nextStats.turns,
        rejection_count: nextStats.rejects,
        tis_score: typeof metrics.TIS === 'number' ? clamp01(metrics.TIS) : undefined,
        metrics: metrics,
        last_message: String(data.content || '').slice(0, 200),
        status: decision === DecisionType.REJECT ? 'rejected' : 'generating',
      });

      addMessage({
        id: `msg-${Date.now()}-${Math.random()}`,
        agentId, agentName: agentId,
        role: data.agent_role, content: data.content,
        timestamp: Date.now(), phase, decision,
      });

      // Update agent visual state
      [...statsMap.keys()].forEach(id => updateAgent(id, { status: 'idle', isThinking: false }));
      updateAgent(agentId, { status: 'speaking', isThinking: true, stats: nextStats });
      setTimeout(() => updateAgent(agentId, { status: 'idle', isThinking: false }), 1200);

      // Metrics
      updateMetrics(prev => ({
        ...prev,
        ldcl: typeof metrics.LDCL === 'number' ? clamp01(metrics.LDCL) : prev.ldcl,
        gne: typeof metrics.GNE === 'number' ? clamp01(metrics.GNE) : prev.gne,
        mlasProbability: typeof metrics.MLAS === 'number' ? clamp01(metrics.MLAS) : prev.mlasProbability,
        diversityCheck: typeof metrics.DIVERSITY === 'boolean' ? metrics.DIVERSITY : prev.diversityCheck,
        softRoleBias: typeof metrics.SOFT_ROLE_BIAS === 'number' ? clamp01(metrics.SOFT_ROLE_BIAS) : prev.softRoleBias,
        tier0: {
          sd: clamp01(metrics.SD) * 100 || prev.tier0.sd,
          rc: clamp01(metrics.RC) * 100 || prev.tier0.rc,
          is: clamp01(metrics.IS) * 100 || prev.tier0.is,
          eic: clamp01(metrics.EIC) * 100 || prev.tier0.eic,
          stability: clamp01(metrics.St) * 100 || prev.tier0.stability,
        },
      }));

      // Chart
      updateChart(prev => {
        const newPoint: MetricPoint = {
          timestamp: now,
          tis: clamp01(metrics.TIS) * 100,
          rc: clamp01(metrics.RC) * 100,
          stability: clamp01(metrics.St) * 100,
          tokenCost: Math.min(100, Math.max(0, (metrics.C || metrics.tok / 1000 || 0) * 100)),
          arDensity: 0,
          accepted: decision === DecisionType.ACCEPT ? 1 : 0,
          rejected: decision === DecisionType.REJECT ? 1 : 0,
          rewrites: decision === DecisionType.REWRITE ? 1 : 0,
          decision, phase,
        };
        const newData = [...prev, newPoint];
        if (newData.length > 40) newData.shift();
        const window = newData.slice(-12);
        const acc = window.filter(p => p.decision === DecisionType.ACCEPT).length;
        const rej = window.filter(p => p.decision === DecisionType.REJECT).length;
        const total = acc + rej;
        newPoint.arDensity = total > 0 ? (acc / total) * 100 : 0;
        return newData;
      });
      return;
    }

    // ── Auto-Score Result ──────────────────────────────────────────────────
    if (data.type === 'auto_score' && data.scores) {
      const s = data.scores;
      // Scores from orchestrator are on 0-10 scale; gauges expect 0-1
      const norm = (v: any, fallback = 0) => {
        const n = parseFloat(v);
        if (isNaN(n)) return fallback;
        return n > 1 ? clamp01(n / 10) : clamp01(n);
      };
      setQualityScores({
        coverage: norm(s.coverage ?? s.Coverage),
        depth: norm(s.depth ?? s.Depth),
        coherence: norm(s.coherence ?? s.Coherence),
        accuracy: norm(s.accuracy ?? s.Accuracy),
        actionability: norm(s.actionability ?? s.Actionability),
        formatting: norm(s.formatting ?? s.Formatting),
      });
      const overall = norm(s.overall ?? s.Overall);
      addEventLog(`[Score] Overall: ${(overall * 10).toFixed(1)}/10`);
      return;
    }

    // ── Synthesized Doc ────────────────────────────────────────────────────
    if (data.type === 'synthesis_complete' && data.document) {
      setSynthesizedDoc(data.document);
      addEventLog('[Synthesis] Document ready');
      return;
    }

    // ── Run Completed ──────────────────────────────────────────────────────
    if (data.type === 'done' || (data.type === 'status' && data.status === 'completed')) {
      setAppPhase(AppPhase.COMPLETED);
      eventSource?.close();
      return;
    }
  };

  eventSource.onerror = () => {
    addEventLog('[Error] SSE connection lost');
  };
};

export const stopSimulation = () => {
  if (eventSource) { eventSource.close(); eventSource = null; }
};

// ── Helpers ────────────────────────────────────────────────────────────────

const toDecision = (value?: string): DecisionType => {
  if (value === 'REJECT') return DecisionType.REJECT;
  if (value === 'REWRITE') return DecisionType.REWRITE;
  return DecisionType.ACCEPT;
};

const toWorkflowPhase = (value?: string): WorkflowPhase | null => {
  if (!value) return null;
  const v = value.toLowerCase();
  if (v.includes('research')) return WorkflowPhase.RESEARCH;
  if (v.includes('analys')) return WorkflowPhase.ANALYSIS;
  if (v.includes('draft')) return WorkflowPhase.DRAFT;
  if (v.includes('review')) return WorkflowPhase.REVIEW;
  return null;
};

const clamp01 = (val: number | undefined): number => {
  if (typeof val !== 'number' || isNaN(val)) return 0;
  return Math.min(1, Math.max(0, val));
};

const formatTimestamp = (): string =>
  new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
