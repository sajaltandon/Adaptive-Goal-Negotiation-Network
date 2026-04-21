import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  AppPhase, WorkflowPhase, DecisionType,
  TeamFormationData, TeamPlanData, SubgoalInfo,
  AgentWorkspaceState, QualityScores, ChartEvent,
  MetricPoint, PhaseMarker, RunMetrics,
} from './types';
import { startSimulation, stopSimulation } from './services/simulation';
import TeamFormationPanel from './components/TeamFormationPanel';
import DAGPanel from './components/DAGPanel';
import WorkspacesPanel from './components/WorkspacesPanel';
import BottomBar from './components/BottomBar';
import EventLog from './components/EventLog';
import ModelSelectionPanel from './components/ModelSelectionPanel';

const API_URL = import.meta.env.VITE_AGNN_API_URL || 'http://localhost:8000';

// ── Placeholder types for ActiveAgent (minimal) ──────────────────────────────
interface ActiveAgent {
  id: string;
  name: string;
  role: string;
  provider: string;
  version: string;
  avatarColor: string;
  capabilities: string[];
  confidence: number;
  stats: { turns: number; accepted: number; rewrites: number; rejects: number };
  isThinking: boolean;
  status: string;
}

const buildAgent = (id: string, model: string): ActiveAgent => ({
  id, name: model, role: 'Unassigned', provider: 'LM Studio', version: 'Auto',
  avatarColor: '#22d3ee', capabilities: ['Autonomous'], confidence: 0.9,
  stats: { turns: 0, accepted: 0, rewrites: 0, rejects: 0 },
  isThinking: false, status: 'idle',
});

const INITIAL_METRICS: RunMetrics = {
  ldcl: 0, gne: 0, mlasProbability: 0, diversityCheck: false, softRoleBias: 0,
  tier0: { sd: 0, rc: 0, is: 0, eic: 0, stability: 0 },
};

const INITIAL_SCORES: QualityScores = {
  coverage: 0, depth: 0, coherence: 0, accuracy: 0, actionability: 0, formatting: 0,
};

// ── Nav sidebar items ─────────────────────────────────────────────────────────
const NAV = [
  { id: 'setup', icon: '⚙', label: 'Setup' },
  { id: 'team', icon: '🤝', label: 'Team Formation' },
  { id: 'dag', icon: '⬡', label: 'Execution DAG' },
  { id: 'workspaces', icon: '⬜', label: 'Workspaces' },
  { id: 'synthesis', icon: '⇢', label: 'Synthesis' },
  { id: 'output', icon: '📄', label: 'Final Output' },
  { id: 'memory', icon: '🗃', label: 'Session Memory' },
];

// ── Sidebar navigation component ─────────────────────────────────────────────
const Sidebar: React.FC<{ active: string; appPhase: AppPhase; onSelect: (id: string) => void }> = ({ active, appPhase, onSelect }) => (
  <nav style={{ width: 60, flexShrink: 0, background: 'rgba(2,6,23,0.9)', borderRight: '1px solid rgba(255,255,255,0.06)', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '12px 0', gap: 4, zIndex: 10 }}>
    {NAV.map((item, i) => {
      const isActive = active === item.id;
      const isEnabled = appPhase !== AppPhase.SETUP || item.id === 'setup';
      return (
        <button key={item.id} title={item.label} onClick={() => isEnabled && onSelect(item.id)} style={{ width: 44, height: 44, borderRadius: 10, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 2, cursor: isEnabled ? 'pointer' : 'default', background: isActive ? 'rgba(34,211,238,0.12)' : 'transparent', border: isActive ? '1px solid #22d3ee33' : '1px solid transparent', opacity: isEnabled ? 1 : 0.25, transition: 'all 0.2s' }}>
          <span style={{ fontSize: 16 }}>{item.icon}</span>
          <span style={{ fontSize: 6, color: isActive ? '#22d3ee' : '#475569', fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.5 }}>{item.label.split(' ')[0]}</span>
        </button>
      );
    })}
  </nav>
);

// ── Main App ──────────────────────────────────────────────────────────────────
const App: React.FC = () => {
  const [appPhase, setAppPhase] = useState<AppPhase>(AppPhase.SETUP);
  const [workflowPhase, setWorkflowPhase] = useState<WorkflowPhase>(WorkflowPhase.RESEARCH);
  const [navTab, setNavTab] = useState<string>('setup');
  const [lmUrl, setLmUrl] = useState('http://10.119.170.167:1234');
  const [prompt, setPrompt] = useState('');
  const [connStatus, setConnStatus] = useState<'idle' | 'ok' | 'err'>('idle');
  const [discoveredModels, setDiscoveredModels] = useState<string[]>([]);
  // dashPhase: 'model_select' = dashboard visible but run not started yet
  //            'running'      = run is in progress
  const [dashPhase, setDashPhase] = useState<'model_select' | 'running'>('model_select');
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [sessionId, setSessionId] = useState('');
  const [elapsed, setElapsed] = useState(0);
  const elapsedRef = useRef<number | null>(null);

  // Agent state (thin wrapper, actual state tracked in workspaces)
  const [agents, setAgents] = useState<ActiveAgent[]>([]);

  // Panel state
  const [teamFormation, setTeamFormation] = useState<TeamFormationData | null>(null);
  const [taskBadge, setTaskBadge] = useState<{ type: string; complexity: string; budget: string } | undefined>();
  const [subgoals, setSubgoals] = useState<SubgoalInfo[]>([]);
  const [workspaces, setWorkspaces] = useState<AgentWorkspaceState[]>([]);
  const [qualityScores, setQualityScores] = useState<QualityScores>(INITIAL_SCORES);
  const [synthesizedDoc, setSynthesizedDoc] = useState('');
  const [showDocModal, setShowDocModal] = useState(false);
  const [synthStatus, setSynthStatus] = useState<'pending' | 'running' | 'done'>('pending');
  const [scoringStatus, setScoringStatus] = useState<'pending' | 'running' | 'done'>('pending');
  const [eventLog, setEventLog] = useState<ChartEvent[]>([]);
  const [chartData] = useState<MetricPoint[]>([]);
  const [phaseMarkers] = useState<PhaseMarker[]>([]);
  const [metrics, setMetrics] = useState<RunMetrics>(INITIAL_METRICS);
  const [semaphoreActive, setSemaphoreActive] = useState(0);

  // ── Elapsed timer ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (appPhase === AppPhase.RUNNING) {
      setElapsed(0);
      elapsedRef.current = window.setInterval(() => setElapsed(e => e + 1), 1000);
    } else {
      if (elapsedRef.current) clearInterval(elapsedRef.current);
    }
    return () => { if (elapsedRef.current) clearInterval(elapsedRef.current); };
  }, [appPhase]);

  // ── Connect to LM Studio ───────────────────────────────────────────────────
  const handleConnect = async () => {
    setConnStatus('idle');
    try {
      const res = await fetch(`${API_URL}/models?base_url=${encodeURIComponent(lmUrl)}`);
      if (!res.ok) throw new Error();
      const data = await res.json();
      setDiscoveredModels(data.models || []);
      setConnStatus('ok');
    } catch {
      setConnStatus('err');
    }
  };

  // ── Helpers ────────────────────────────────────────────────────────────────
  const updateAgent = useCallback((id: string, updates: Partial<ActiveAgent>) => {
    setAgents(prev => {
      const idx = prev.findIndex(a => a.id === id);
      if (idx >= 0) {
        const next = [...prev];
        next[idx] = { ...next[idx], ...updates } as ActiveAgent;
        return next;
      }
      return [...prev, { ...buildAgent(id, id), ...updates } as ActiveAgent];
    });
  }, []);

  const updateWorkspace = useCallback((id: string, updates: Partial<AgentWorkspaceState>) => {
    setWorkspaces(prev => {
      const idx = prev.findIndex(w => w.agent_id === id);
      if (idx >= 0) {
        const next = [...prev];
        next[idx] = { ...next[idx], ...updates };
        return next;
      }
      if ((updates as any).model !== undefined || (updates as any).primary_role !== undefined) {
        return [...prev, { agent_id: id, model: '', primary_role: '', secondary_roles: [], turn_count: 0, max_turns: 12, tis_score: 0, rejection_count: 0, last_message: '', status: 'pending', elapsed_s: 0, ...updates }];
      }
      return prev;
    });
    // Update semaphore
    setWorkspaces(ws => {
      const active = ws.filter(w => w.status === 'generating' || w.status === 'hot-swapping').length;
      setSemaphoreActive(active);
      return ws;
    });
  }, []);

  const addEventLog = useCallback((label: string) => {
    const ts = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const kind: ChartEvent['kind'] = label.startsWith('[⚡') ? 'hotswap' : label.startsWith('[Phase]') ? 'phase' : label.startsWith('[Error]') ? 'error' : 'system';
    setEventLog(prev => [...prev.slice(-100), { timestamp: ts, label, kind }]);
  }, []);

  // ── Go to Dashboard (from Launch on setup page) ───────────────────────────
  const handleGoToDashboard = () => {
    if (!prompt.trim() || connStatus !== 'ok') return;
    setAppPhase(AppPhase.RUNNING);   // show dashboard
    setDashPhase('model_select');     // but wait for model selection
    setNavTab('team');
    setTeamFormation(null); setSubgoals([]); setWorkspaces([]);
    setQualityScores(INITIAL_SCORES); setSynthesizedDoc('');
    setSynthStatus('pending'); setScoringStatus('pending');
    setEventLog([]); setAgents([]);
    const sid = `AGNN-${Date.now().toString(36).toUpperCase()}`;
    setSessionId(sid);
  };

  // ── Start Run (called by ModelSelectionPanel after model confirmation) ────
  const handleStart = async (models?: string[]) => {
    if (!prompt.trim()) return;
    setDashPhase('running');
    if (models) setSelectedModels(models);

    try {
      await startSimulation(
        models || selectedModels,
        prompt,
        lmUrl,
        true, // <--- enableTier2 must be true for the DAG to run!
        (_msg) => { /* messages via eventLog */ },
        updateAgent,
        (fn) => setMetrics(fn),
        (_fn) => { /* chart */ },
        (_fn) => { /* phase markers */ },
        (fn) => setEventLog(prev => fn(prev) as ChartEvent[]),
        (fn) => setTeamFormation(fn as any),
        (_plan) => { /* team plan */ },
        setAppPhase,
        setWorkflowPhase,
        (fn) => setSubgoals(fn),
        updateWorkspace,
        (scores) => { setQualityScores(scores); setScoringStatus('done'); },
        (doc) => { setSynthesizedDoc(doc); setSynthStatus('done'); setNavTab('output'); },
        addEventLog,
      );
    } catch (e: any) {
      addEventLog(`[Error] Failed to start: ${e?.message || e}`);
      setAppPhase(AppPhase.SETUP);
    }
  };

  // ── Handle task analysis events from event log ────────────────────────────
  useEffect(() => {
    const lastAnalysis = eventLog.filter(e => e.label.startsWith('[Analysis]')).slice(-1)[0];
    if (lastAnalysis) {
      const m = lastAnalysis.label.match(/Type: (\S+) \| Complexity: (\S+) \| Team: (\d+)/);
      if (m) setTaskBadge({ type: m[1], complexity: m[2], budget: 'quality_first' });
    }
    // Check synthesis events
    if (eventLog.some(e => e.label === '[Synthesis] Document ready')) setSynthStatus('done');
    else if (eventLog.some(e => e.label.includes('[Score]'))) setScoringStatus('done');
  }, [eventLog]);

  // ── Reset ──────────────────────────────────────────────────────────────────
  const handleReset = () => {
    stopSimulation();
    setAppPhase(AppPhase.SETUP);
    setNavTab('setup');
    setDashPhase('model_select');
    setSelectedModels([]);
    setTeamFormation(null); setSubgoals([]); setWorkspaces([]); setAgents([]);
    setQualityScores(INITIAL_SCORES); setSynthesizedDoc('');
    setSynthStatus('pending'); setScoringStatus('pending');
    setEventLog([]); setSessionId(''); setElapsed(0);
  };

  // ── Download synthesized doc ───────────────────────────────────────────────
  const handleDownload = () => {
    if (!synthesizedDoc) return;
    const blob = new Blob([synthesizedDoc], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `agnn_output_${sessionId || Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── Elapsed formatter ──────────────────────────────────────────────────────
  const fmtElapsed = `${String(Math.floor(elapsed / 3600)).padStart(2, '0')}:${String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0')}:${String(elapsed % 60).padStart(2, '0')}`;

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', overflow: 'hidden', background: '#030712', color: '#e2e8f0', fontFamily: "'Inter', system-ui, sans-serif" }}>

      {/* ── Global style injections ────────────────────────────────────────── */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
        @keyframes shimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(200%); } }
      `}</style>

      {/* ── Top bar ─────────────────────────────────────────────────────────── */}
      <header style={{ height: 48, flexShrink: 0, display: 'flex', alignItems: 'center', padding: '0 16px', gap: 14, background: 'rgba(2,6,23,0.95)', borderBottom: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(16px)', zIndex: 50 }}>
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 28, height: 28, borderRadius: 8, background: 'linear-gradient(135deg,#22d3ee33,#818cf833)', border: '1px solid #22d3ee44', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14, fontWeight: 800, color: '#22d3ee' }}>A</div>
          <span style={{ fontSize: 14, fontWeight: 800, letterSpacing: '-0.5px', color: '#f8fafc' }}>AGNN</span>
        </div>

        <div style={{ width: 1, height: 20, background: 'rgba(255,255,255,0.08)' }} />

        {/* Connection status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 5, background: connStatus === 'ok' ? 'rgba(52,211,153,0.1)' : 'rgba(71,85,105,0.2)', border: `1px solid ${connStatus === 'ok' ? '#34d39933' : 'rgba(71,85,105,0.4)'}`, borderRadius: 20, padding: '3px 10px' }}>
          <div style={{ width: 6, height: 6, borderRadius: '50%', background: connStatus === 'ok' ? '#34d399' : connStatus === 'err' ? '#f87171' : '#475569', boxShadow: connStatus === 'ok' ? '0 0 6px #34d399' : 'none' }} />
          <span style={{ fontSize: 10, fontWeight: 600, color: connStatus === 'ok' ? '#34d399' : '#64748b' }}>LM Studio {connStatus === 'ok' ? 'connected' : connStatus === 'err' ? 'error' : 'disconnected'}</span>
        </div>

        {sessionId && (
          <div style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 6, padding: '3px 10px', fontSize: 10, color: '#475569', fontFamily: 'monospace' }}>
            SESSION_ID: {sessionId}
          </div>
        )}

        {appPhase === AppPhase.RUNNING && (
          <div style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 6, padding: '3px 10px', fontSize: 10, color: '#94a3b8', fontFamily: 'monospace' }}>
            ELAPSED: {fmtElapsed}
          </div>
        )}

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button onClick={handleReset} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '5px 12px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 7, cursor: 'pointer', fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>
            ↺ New Run
          </button>
          <button onClick={handleDownload} disabled={!synthesizedDoc} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '5px 12px', background: synthesizedDoc ? 'rgba(34,211,238,0.12)' : 'rgba(255,255,255,0.03)', border: `1px solid ${synthesizedDoc ? '#22d3ee33' : 'rgba(255,255,255,0.06)'}`, borderRadius: 7, cursor: synthesizedDoc ? 'pointer' : 'default', fontSize: 11, color: synthesizedDoc ? '#22d3ee' : '#334155', fontWeight: 600 }}>
            ↓ Export
          </button>
          <button style={{ padding: '5px 12px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 7, cursor: 'pointer', fontSize: 11, color: '#64748b', fontWeight: 600 }}>
            ⚙ Settings
          </button>
        </div>
      </header>

      {/* ── Main layout ─────────────────────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* Sidebar */}
        <Sidebar active={navTab} appPhase={appPhase} onSelect={setNavTab} />

        {/* Content area */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

          {/* ── SETUP panel ───────────────────────────────────────────── */}
          {navTab === 'setup' && (
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 40 }}>
              <div style={{ width: '100%', maxWidth: 560, display: 'flex', flexDirection: 'column', gap: 18 }}>

                <div>
                  <h2 style={{ fontSize: 22, fontWeight: 800, color: '#f8fafc', letterSpacing: '-0.5px' }}>Configure AGNN Run</h2>
                  <p style={{ fontSize: 13, color: '#475569', marginTop: 4 }}>Connect to model hub (LM Studio + optional Gemini/Groq) and define your task prompt</p>
                </div>

                {/* LM Studio URL */}
                <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12, padding: '16px 20px' }}>
                  <label style={{ fontSize: 11, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: 1, display: 'block', marginBottom: 8 }}>LM Studio URL</label>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <input value={lmUrl} onChange={e => setLmUrl(e.target.value)} style={{ flex: 1, background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, padding: '8px 12px', color: '#e2e8f0', fontSize: 13, fontFamily: 'monospace', outline: 'none' }} placeholder="http://..." />
                    <button onClick={handleConnect} style={{ padding: '8px 18px', background: 'rgba(34,211,238,0.12)', border: '1px solid #22d3ee33', borderRadius: 8, cursor: 'pointer', fontSize: 12, color: '#22d3ee', fontWeight: 700 }}>Connect</button>
                  </div>
                  {connStatus === 'ok' && <div style={{ marginTop: 8, fontSize: 11, color: '#34d399' }}>✓ Connected — {discoveredModels.length} model(s) found across LM Studio/Gemini/Groq</div>}
                  {connStatus === 'err' && <div style={{ marginTop: 8, fontSize: 11, color: '#f87171' }}>✗ Could not reach LM Studio.</div>}
                </div>

                {/* Prompt */}
                <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12, padding: '16px 20px' }}>
                  <label style={{ fontSize: 11, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: 1, display: 'block', marginBottom: 8 }}>Task Prompt</label>
                  <textarea value={prompt} onChange={e => setPrompt(e.target.value)} rows={5} style={{ width: '100%', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, padding: '10px 12px', color: '#e2e8f0', fontSize: 13, resize: 'vertical', outline: 'none', fontFamily: 'inherit', lineHeight: 1.6 }} placeholder="Describe the task for the AGNN agent team..." />
                </div>

                <button
                  onClick={handleGoToDashboard}
                  disabled={!prompt.trim() || connStatus !== 'ok'}
                  style={{ padding: '14px', background: connStatus === 'ok' && prompt.trim() ? 'linear-gradient(135deg,#22d3ee22,#818cf833)' : 'rgba(255,255,255,0.04)', border: `1px solid ${connStatus === 'ok' && prompt.trim() ? '#22d3ee55' : 'rgba(255,255,255,0.07)'}`, borderRadius: 12, cursor: connStatus === 'ok' && prompt.trim() ? 'pointer' : 'default', fontSize: 15, color: connStatus === 'ok' && prompt.trim() ? '#22d3ee' : '#334155', fontWeight: 800, letterSpacing: '-0.3px', transition: 'all 0.3s', boxShadow: connStatus === 'ok' && prompt.trim() ? '0 0 20px #22d3ee11' : 'none' }}>
                  → Go to Dashboard
                </button>
              </div>
            </div>
          )}

          {/* ── RUNNING dashboard ─────────────────────────────────────────── */}
          {navTab !== 'setup' && navTab !== 'output' && navTab !== 'memory' && (
            <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gridTemplateRows: '1fr', gap: 0, overflow: 'hidden' }}>

              {/* Column 1: Model Selection + Team Formation */}
              <div style={{ borderRight: '1px solid rgba(255,255,255,0.06)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

                {/* ── Model Selection Panel (top of col 1) ── */}
                <div style={{ flexShrink: 0, borderBottom: '1px solid rgba(255,255,255,0.06)', display: 'flex', flexDirection: 'column', overflow: 'hidden', maxHeight: dashPhase === 'model_select' ? '60%' : 200, transition: 'max-height 0.5s ease', minHeight: 120 }}>
                  <div style={{ padding: '10px 14px', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
                    <span style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8', letterSpacing: 0.5 }}>Model Selection</span>
                    <span style={{ fontSize: 9, padding: '2px 6px', borderRadius: 4, background: dashPhase === 'running' ? 'rgba(52,211,153,0.12)' : 'rgba(251,146,60,0.12)', color: dashPhase === 'running' ? '#34d399' : '#fb923c', fontWeight: 700 }}>
                      {dashPhase === 'running' ? 'CONFIRMED' : 'PENDING'}
                    </span>
                  </div>
                  <div style={{ flex: 1, overflow: 'auto', padding: '10px 14px' }}>
                    <ModelSelectionPanel
                      discoveredModels={discoveredModels}
                      prompt={prompt}
                      apiUrl={API_URL}
                      lmUrl={lmUrl}
                      onStartRun={handleStart}
                    />
                  </div>
                </div>

                {/* ── Tier-1: Team Formation (bottom of col 1) ── */}
                <div style={{ padding: '10px 14px', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
                  <span style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8', letterSpacing: 0.5 }}>Tier-1: Team Formation</span>
                  <span style={{ fontSize: 9, color: '#334155' }}>···</span>
                </div>
                <div style={{ flex: 1, overflow: 'hidden', padding: '10px 14px' }}>
                  <TeamFormationPanel data={teamFormation} taskBadge={taskBadge} />
                </div>
              </div>

              {/* Column 2: DAG Execution */}
              <div style={{ borderRight: '1px solid rgba(255,255,255,0.06)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                <div style={{ padding: '10px 14px', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
                  <span style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8', letterSpacing: 0.5 }}>Tier-2: DAG Execution</span>
                  <span style={{ fontSize: 9, color: '#334155' }}>···</span>
                </div>
                <div style={{ flex: 1, overflow: 'auto', padding: '10px 14px' }}>
                  <DAGPanel subgoals={subgoals} semaphoreSlots={4} semaphoreActive={semaphoreActive} />
                </div>
              </div>

              {/* Column 3: Workspaces + Event log */}
              <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                {/* Workspaces */}
                <div style={{ flex: 3, borderBottom: '1px solid rgba(255,255,255,0.06)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                  <div style={{ padding: '10px 14px', borderBottom: '1px solid rgba(255,255,255,0.05)', flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8', letterSpacing: 0.5 }}>Live Agent Workspaces</span>
                    <span style={{ fontSize: 9, color: '#334155' }}>···</span>
                  </div>
                  <div style={{ flex: 1, overflow: 'auto', padding: '10px 14px' }}>
                    <WorkspacesPanel workspaces={workspaces} />
                  </div>
                </div>

                {/* Event log */}
                <div style={{ flex: 2, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                  <div style={{ padding: '8px 14px', borderBottom: '1px solid rgba(255,255,255,0.05)', flexShrink: 0 }}>
                    <span style={{ fontSize: 11, fontWeight: 700, color: '#94a3b8', letterSpacing: 0.5 }}>Event Log</span>
                  </div>
                  <div style={{ flex: 1, overflow: 'hidden', padding: '8px 14px' }}>
                    <EventLog events={eventLog} sessionId={sessionId} elapsed={elapsed} />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ── Output panel ────────────────────────────────────────────────── */}
          {navTab === 'output' && (
            <div style={{ flex: 1, overflow: 'auto', padding: 32 }}>
              <h2 style={{ fontSize: 18, fontWeight: 800, color: '#f8fafc', marginBottom: 16 }}>Final Document</h2>
              {synthesizedDoc ? (
                <pre style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 12, padding: 24, fontSize: 13, color: '#94a3b8', whiteSpace: 'pre-wrap', lineHeight: 1.8, fontFamily: 'inherit' }}>
                  {synthesizedDoc}
                </pre>
              ) : (
                <div style={{ color: '#334155', fontSize: 13 }}>No document synthesized yet. Run AGNN to completion first.</div>
              )}
            </div>
          )}

          {/* ── Bottom bar ──────────────────────────────────────────────────── */}
          {appPhase !== AppPhase.SETUP && (
            <div style={{ flexShrink: 0 }}>
              <BottomBar
                synthStatus={synthStatus}
                scoringStatus={scoringStatus}
                outputReady={!!synthesizedDoc}
                scores={qualityScores}
                onViewDoc={() => setNavTab('output')}
                onDownload={handleDownload}
              />
            </div>
          )}
        </div>
      </div>

      {/* ── Doc modal ─────────────────────────────────────────────────────── */}
      {showDocModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.8)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 32 }} onClick={() => setShowDocModal(false)}>
          <div style={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 16, width: '100%', maxWidth: 800, maxHeight: '80vh', overflow: 'auto', padding: 32 }} onClick={e => e.stopPropagation()}>
            <pre style={{ fontSize: 13, color: '#94a3b8', whiteSpace: 'pre-wrap', lineHeight: 1.8, fontFamily: 'inherit' }}>{synthesizedDoc}</pre>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
