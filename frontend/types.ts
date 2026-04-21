// ─── App-level enums ────────────────────────────────────────────────────────

export enum AppPhase {
  SETUP = 'SETUP',
  NEGOTIATION = 'NEGOTIATION',
  RUNNING = 'RUNNING',
  COMPLETED = 'COMPLETED',
}

export enum WorkflowPhase {
  RESEARCH = 'Research',
  ANALYSIS = 'Analysis',
  DRAFT = 'Draft',
  REVIEW = 'Review',
}

export enum DecisionType {
  ACCEPT = 'ACCEPT',
  REWRITE = 'REWRITE',
  REJECT = 'REJECT',
  INFO = 'INFO',
}

export enum AgentRole {
  RESEARCHER = 'Researcher',
  CRITIC = 'Critic',
  DRAFTER = 'Drafter',
  REVIEWER = 'Reviewer',
  GENERALIST = 'Generalist',
}

// ─── Agent / Team types ─────────────────────────────────────────────────────

export interface AgentModel {
  id: string;
  name: string;
  provider: string;
  version: string;
  avatarColor: string;
  capabilities: string[];
}

export interface ActiveAgent {
  id: string;
  name: string;
  role: AgentRole | string;
  provider: string;
  version: string;
  avatarColor: string;
  capabilities: string[];
  confidence: number;
  stats: {
    turns: number;
    accepted: number;
    rewrites: number;
    rejects: number;
  };
  isThinking: boolean;
  status: string;
  autoAdded?: boolean;
}

export interface Message {
  id: string;
  agentId: string;
  agentName?: string;
  role?: string;
  content: string;
  timestamp: number;
  phase?: WorkflowPhase | string;
  decision?: DecisionType;
  isPhaseChange?: boolean;
}

export interface RoleDefinition {
  name: string;
  description: string;
  responsibilities: string[];
  system_prompt_addition?: string;
}

export interface TeamMemberInfo {
  agent_id: string;
  model: string;
  role: RoleDefinition;
  confidence: number;
  secondary_roles?: RoleDefinition[];
  capabilities?: string[];
}

export interface NegotiationMessage {
  turn: number;
  agent: string;
  content: string;
  proposed_role?: string;
  final_role?: string;
  negotiation_score?: number;
  phase?: 'primary' | 'secondary';
}

export interface TeamFormationData {
  members: TeamMemberInfo[];
  negotiation: NegotiationMessage[];
  formationTurns: number;
  consensusStrength?: number;
  roles?: RoleDefinition[];
}

export interface TeamPlanData {
  responsibilities: Record<string, string[]>;
  avoid: Record<string, string[]>;
  turn_order?: Record<string, string[]>;
}

// ─── Subgoal / DAG types ────────────────────────────────────────────────────

export type SubgoalStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface SubgoalInfo {
  id: number;
  name: string;
  description: string;
  phase_type: string;   // research | analysis | draft | review
  dependencies: number[];
  assigned_agent?: string;  // Agent ID
  assigned_role?: string;   // Role name
  status: SubgoalStatus;
  turns?: number;
  max_turns?: number;
}

// ─── Workspace (per-agent live view) ────────────────────────────────────────

export type WorkspaceStatus = 'pending' | 'generating' | 'accepted' | 'rejected' | 'hot-swapping' | 'done';

export interface AgentWorkspaceState {
  agent_id: string;
  model: string;
  primary_role: string;
  secondary_roles: string[];
  current_subgoal?: string;
  subgoal_phase?: string;
  turn_count: number;
  max_turns: number;
  tis_score: number;
  metrics?: Record<string, number>;
  rejection_count: number;
  last_message: string;
  status: WorkspaceStatus;
  elapsed_s: number;
}

// ─── Metrics / Chart ────────────────────────────────────────────────────────

export interface MetricPoint {
  timestamp: string;
  tis: number;
  rc: number;
  stability: number;
  tokenCost: number;
  arDensity: number;
  accepted: number;
  rejected: number;
  rewrites: number;
  decision?: DecisionType;
  phase?: string;
}

export interface PhaseMarker {
  timestamp: string;
  label: string;
}

export interface ChartEvent {
  timestamp: string;
  label: string;
  kind: 'phase' | 'role' | 'plan' | 'system' | 'error' | 'hotswap';
}

export interface QualityScores {
  coverage: number;
  depth: number;
  coherence: number;
  accuracy: number;
  actionability: number;
  formatting: number;
}

export interface RunMetrics {
  ldcl: number;
  gne: number;
  mlasProbability: number;
  diversityCheck: boolean;
  softRoleBias: number;
  tier0: {
    sd: number;
    rc: number;
    is: number;
    eic: number;
    stability: number;
  };
}
