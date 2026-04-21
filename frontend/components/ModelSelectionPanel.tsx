import React, { useState, useEffect } from 'react';

interface RankedModel { model: string; score: number; selected: boolean; }
interface ReliabilityResult { model: string; status: 'checking' | 'ok' | 'fail'; latency?: number; error?: string; }
interface TaskAnalysis {
    task_type: string; complexity: string;
    team_size: number; budget: string; sub_steps: string[];
}

interface Props {
    discoveredModels: string[];
    prompt: string;
    apiUrl: string;
    lmUrl: string;
    onStartRun: (selectedModels: string[]) => void;
}

// ── helpers ──────────────────────────────────────────────────────────────────

const Spinner: React.FC<{ sz?: number; color?: string }> = ({ sz = 12, color = '#22d3ee' }) => (
    <span style={{ display: 'inline-block', width: sz, height: sz, borderRadius: '50%', border: `2px solid ${color}`, borderTopColor: 'transparent', animation: 'spin 0.8s linear infinite', flexShrink: 0 }} />
);

const Badge: React.FC<{ label: string; value: string; color: string }> = ({ label, value, color }) => (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', background: `${color}12`, border: `1px solid ${color}33`, borderRadius: 8, padding: '5px 9px', minWidth: 68 }}>
        <span style={{ fontSize: 8, color: '#475569', fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.8, marginBottom: 2 }}>{label}</span>
        <span style={{ fontSize: 11, fontWeight: 800, color, letterSpacing: '-0.2px', textAlign: 'center', lineHeight: 1.2 }}>{value}</span>
    </div>
);

const cxColor = (c: string) =>
    /high|complex/i.test(c) ? '#f87171' : /med/i.test(c) ? '#fb923c' : '#34d399';

const providerForModel = (model: string): 'Gemini' | 'Groq' | 'LM Studio' => {
    if (model.startsWith('models/gemini')) return 'Gemini';
    if (model.startsWith('groq/')) return 'Groq';
    return 'LM Studio';
};

const providerTint = (provider: 'Gemini' | 'Groq' | 'LM Studio') => {
    if (provider === 'Gemini') return '#818cf8';
    if (provider === 'Groq') return '#f472b6';
    return '#22d3ee';
};

// ── component ────────────────────────────────────────────────────────────────

const ModelSelectionPanel: React.FC<Props> = ({ discoveredModels, prompt, apiUrl, lmUrl, onStartRun }) => {

    // ── Task analysis state (auto-runs on mount, re-runs on retry) ──────────
    type AnalysisPhase = 'loading' | 'done' | 'error';
    const [analysisPhase, setAnalysisPhase] = useState<AnalysisPhase>('loading');
    const [taskAnalysis, setTaskAnalysis] = useState<TaskAnalysis | null>(null);
    const [ranked, setRanked] = useState<RankedModel[]>([]);
    const [analysisError, setAnalysisError] = useState('');
    const [retryKey, setRetryKey] = useState(0);   // increment to retry

    useEffect(() => {
        // Auto-run task analysis + model ranking immediately on mount (or retry)
        let cancelled = false;
        setAnalysisPhase('loading');
        setTaskAnalysis(null);
        setRanked([]);
        const run = async () => {
            try {
                const res = await fetch(`${apiUrl}/rank-models`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ base_url: lmUrl, prompt, max_agents: 4 }),
                });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                if (cancelled) return;
                setTaskAnalysis({
                    task_type: data.task_type || 'general',
                    complexity: data.complexity || 'unknown',
                    team_size: data.team_size || data.selected_models?.length || 3,
                    budget: data.budget || 'quality_first',
                    sub_steps: data.sub_steps || [],
                });
                setRanked(data.ranked_models || []);
                setAnalysisPhase('done');
            } catch (e: any) {
                if (cancelled) return;
                setAnalysisError(e?.message || 'Backend unreachable');
                setAnalysisPhase('error');
            }
        };
        run();
        return () => { cancelled = true; };
    }, [retryKey]);

    // ── Model selection ──────────────────────────────────────────────────────
    const [mode, setMode] = useState<'auto' | 'manual'>('auto');
    const [manualChecked, setManualChecked] = useState<Set<string>>(
        new Set(discoveredModels.slice(0, 3))
    );
    useEffect(() => {
        setManualChecked(new Set(discoveredModels.slice(0, 3)));
    }, [discoveredModels]);
    const toggleManual = (m: string) => setManualChecked(prev => {
        const n = new Set(prev); n.has(m) ? n.delete(m) : n.add(m); return n;
    });

    // ── Reliability check ────────────────────────────────────────────────────
    const [reliPhase, setReliPhase] = useState<'idle' | 'checking' | 'done'>('idle');
    const [reliResults, setReliResults] = useState<ReliabilityResult[]>([]);

    const runReliability = (models: string[]) => {
        setReliPhase('checking');
        setReliResults(models.map(m => ({ model: m, status: 'checking' })));
        const probe = async (idx: number) => {
            if (idx >= models.length) { setReliPhase('done'); return; }
            const model = models[idx];
            try {
                const r = await fetch(`${apiUrl}/probe-model`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ base_url: lmUrl, model }),
                });
                const d = await r.json();
                setReliResults(prev => prev.map(x => x.model === model
                    ? { ...x, status: d.ok ? 'ok' : 'fail', latency: d.latency_ms, error: d.error }
                    : x));
            } catch {
                setReliResults(prev => prev.map(x => x.model === model
                    ? { ...x, status: 'fail', error: 'Network error' } : x));
            }
            probe(idx + 1);
        };
        probe(0);
    };

    const selectedList = mode === 'auto'
        ? ranked.filter(r => r.selected).map(r => r.model)
        : Array.from(manualChecked);

    // ── Render ───────────────────────────────────────────────────────────────
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, height: '100%', overflowY: 'auto' }}>

            {/* ── Phase 1: Task Analysis (automatic) ── */}
            <div style={{ flexShrink: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                    {analysisPhase === 'loading' ? <Spinner sz={11} /> : analysisPhase === 'done' ? <span style={{ fontSize: 11, color: '#34d399' }}>✓</span> : <span style={{ fontSize: 11, color: '#f87171' }}>✗</span>}
                    <span style={{ fontSize: 10, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.8 }}>
                        {analysisPhase === 'loading' ? 'Analyzing task…' : analysisPhase === 'done' ? 'Task Analysis' : 'Analysis Failed'}
                    </span>
                </div>

                {analysisPhase === 'loading' && (
                    <div style={{ fontSize: 10, color: '#334155', paddingLeft: 17 }}>
                        Detecting type · complexity · team size · budget…
                    </div>
                )}

                {analysisPhase === 'error' && (
                    <div style={{ fontSize: 10, color: '#f87171', padding: '6px 10px', background: 'rgba(248,113,113,0.08)', borderRadius: 7, border: '1px solid #f8717133' }}>
                        ⚠ {analysisError}
                        <button onClick={() => setRetryKey(k => k + 1)}
                            style={{ display: 'block', marginTop: 4, padding: '2px 8px', background: 'rgba(248,113,113,0.15)', border: '1px solid #f8717133', borderRadius: 4, cursor: 'pointer', color: '#f87171', fontSize: 9 }}>
                            Retry
                        </button>
                    </div>
                )}

                {analysisPhase === 'done' && taskAnalysis && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                        {/* Badges row */}
                        <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
                            <Badge label="Task Type" value={taskAnalysis.task_type} color="#818cf8" />
                            <Badge label="Complexity" value={taskAnalysis.complexity} color={cxColor(taskAnalysis.complexity)} />
                            <Badge label="Team Size" value={`${taskAnalysis.team_size} agents`} color="#22d3ee" />
                            <Badge label="Budget" value={taskAnalysis.budget.replace(/_/g, ' ')} color="#fb923c" />
                        </div>
                        {/* Sub-steps */}
                        {taskAnalysis.sub_steps.length > 0 && (
                            <div style={{ fontSize: 9, color: '#334155', display: 'flex', flexWrap: 'wrap', gap: 3 }}>
                                {taskAnalysis.sub_steps.slice(0, 5).map((s, i) => (
                                    <React.Fragment key={i}>
                                        {i > 0 && <span style={{ color: '#1e293b' }}>→</span>}
                                        <span style={{ color: '#475569' }}>{s}</span>
                                    </React.Fragment>
                                ))}
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Divider — only show after analysis done */}
            {analysisPhase === 'done' && (
                <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', flexShrink: 0 }} />
            )}

            {/* ── Phase 2: Model Selection (shown after analysis) ── */}
            {analysisPhase === 'done' && (
                <>
                    {/* Mode toggle */}
                    <div style={{ flexShrink: 0 }}>
                        <div style={{ fontSize: 9, fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: 0.8, marginBottom: 5 }}>
                            Model Selection
                        </div>
                        <div style={{ background: 'rgba(0,0,0,0.4)', borderRadius: 8, padding: 3, display: 'flex', gap: 2 }}>
                            {(['auto', 'manual'] as const).map(m => (
                                <button key={m} onClick={() => { setMode(m); setReliPhase('idle'); setReliResults([]); }}
                                    style={{ flex: 1, padding: '6px 0', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 11, fontWeight: 700, transition: 'all 0.25s', background: mode === m ? 'linear-gradient(135deg,#22d3ee1a,#818cf81a)' : 'transparent', color: mode === m ? '#22d3ee' : '#475569', boxShadow: mode === m ? 'inset 0 0 0 1px #22d3ee33' : 'none' }}>
                                    {m === 'auto' ? '⚡ Auto' : '☑ Manual'}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Auto: pre-ranked list */}
                    {mode === 'auto' && ranked.length > 0 && (
                        <div style={{ flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 4 }}>
                            {ranked.map((r, i) => (
                                <div key={r.model} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '5px 8px', background: r.selected ? 'rgba(34,211,238,0.06)' : 'rgba(255,255,255,0.02)', border: `1px solid ${r.selected ? '#22d3ee33' : 'rgba(255,255,255,0.05)'}`, borderRadius: 7 }}>
                                    <span style={{ fontSize: 9, color: r.selected ? '#22d3ee' : '#334155', fontWeight: 800, minWidth: 16 }}>#{i + 1}</span>
                                    <span style={{ flex: 1, fontSize: 9, color: r.selected ? '#e2e8f0' : '#475569', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.model}</span>
                                    <span style={{
                                        fontSize: 8,
                                        fontWeight: 700,
                                        color: providerTint(providerForModel(r.model)),
                                        background: `${providerTint(providerForModel(r.model))}18`,
                                        border: `1px solid ${providerTint(providerForModel(r.model))}44`,
                                        borderRadius: 4,
                                        padding: '1px 5px',
                                    }}>
                                        {providerForModel(r.model)}
                                    </span>
                                    <div style={{ width: 32, height: 2, background: 'rgba(255,255,255,0.06)', borderRadius: 1, overflow: 'hidden' }}>
                                        <div style={{ height: '100%', width: `${r.score * 100}%`, background: r.selected ? '#22d3ee' : '#334155', borderRadius: 1 }} />
                                    </div>
                                    <span style={{ fontSize: 9, color: r.selected ? '#22d3ee' : '#334155', minWidth: 26, fontWeight: 700 }}>{r.score.toFixed(2)}</span>
                                    {r.selected && <span style={{ fontSize: 9, color: '#22d3ee' }}>✓</span>}
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Manual: checkboxes */}
                    {mode === 'manual' && (
                        <div style={{ flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 4 }}>
                            <div style={{ fontSize: 9, color: '#475569' }}>{manualChecked.size} of {discoveredModels.length} selected</div>
                            {discoveredModels.map(m => {
                                const checked = manualChecked.has(m);
                                return (
                                    <label key={m} style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '5px 8px', background: checked ? 'rgba(34,211,238,0.06)' : 'rgba(255,255,255,0.02)', border: `1px solid ${checked ? '#22d3ee44' : 'rgba(255,255,255,0.06)'}`, borderRadius: 7, cursor: 'pointer', transition: 'all 0.2s' }}>
                                        <div style={{ width: 14, height: 14, borderRadius: 4, border: `2px solid ${checked ? '#22d3ee' : '#334155'}`, background: checked ? '#22d3ee' : 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                                            {checked && <span style={{ fontSize: 8, color: '#030712', fontWeight: 900 }}>✓</span>}
                                        </div>
                                        <input type="checkbox" checked={checked} onChange={() => toggleManual(m)} style={{ display: 'none' }} />
                                        <span style={{ flex: 1, fontSize: 9, color: checked ? '#e2e8f0' : '#64748b', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m}</span>
                                        <span style={{
                                            fontSize: 8,
                                            fontWeight: 700,
                                            color: providerTint(providerForModel(m)),
                                            background: `${providerTint(providerForModel(m))}18`,
                                            border: `1px solid ${providerTint(providerForModel(m))}44`,
                                            borderRadius: 4,
                                            padding: '1px 5px',
                                        }}>
                                            {providerForModel(m)}
                                        </span>
                                    </label>
                                );
                            })}
                        </div>
                    )}

                    {/* ── Phase 3: Reliability check ── */}
                    {reliPhase === 'idle' && selectedList.length > 0 && (
                        <button onClick={() => runReliability(selectedList)}
                            style={{ flexShrink: 0, width: '100%', padding: '7px', background: 'rgba(129,140,248,0.1)', border: '1px solid #818cf844', borderRadius: 8, cursor: 'pointer', color: '#818cf8', fontSize: 11, fontWeight: 700 }}>
                            → Reliability Check ({selectedList.length} models)
                        </button>
                    )}

                    {(reliPhase === 'checking' || reliPhase === 'done') && (
                        <div style={{ flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 4 }}>
                            <div style={{ fontSize: 9, fontWeight: 700, color: '#475569', textTransform: 'uppercase', letterSpacing: 0.7 }}>
                                Reliability {reliPhase === 'checking' ? <Spinner sz={9} color="#22d3ee" /> : '✓'}
                            </div>
                            {reliResults.map(r => (
                                <div key={r.model} style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '5px 8px', background: r.status === 'ok' ? 'rgba(52,211,153,0.05)' : r.status === 'fail' ? 'rgba(248,113,113,0.05)' : 'rgba(255,255,255,0.02)', border: `1px solid ${r.status === 'ok' ? '#34d39933' : r.status === 'fail' ? '#f8717133' : 'rgba(255,255,255,0.06)'}`, borderRadius: 7, transition: 'all 0.3s' }}>
                                    {r.status === 'checking' && <Spinner sz={10} />}
                                    {r.status === 'ok' && <span style={{ fontSize: 10 }}>✅</span>}
                                    {r.status === 'fail' && <span style={{ fontSize: 10 }}>❌</span>}
                                    <span style={{ flex: 1, fontSize: 9, color: '#94a3b8', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.model}</span>
                                    {r.status === 'ok' && <span style={{ fontSize: 9, color: '#34d399', fontWeight: 700 }}>{r.latency}ms</span>}
                                    {r.status === 'fail' && <span style={{ fontSize: 9, color: '#f87171' }}>fail</span>}
                                </div>
                            ))}
                        </div>
                    )}

                    {/* ── Launch ── */}
                    {reliPhase === 'done' && (
                        <button onClick={() => onStartRun(selectedList)}
                            style={{ flexShrink: 0, width: '100%', padding: '10px', background: 'linear-gradient(135deg,#22d3ee22,#818cf833)', border: '1px solid #22d3ee55', borderRadius: 10, cursor: 'pointer', fontSize: 13, color: '#22d3ee', fontWeight: 800, boxShadow: '0 0 20px #22d3ee11', letterSpacing: '-0.3px' }}>
                            ▶ Start AGNN Negotiation
                        </button>
                    )}

                    {/* Quick launch (skip reliability) */}
                    {reliPhase === 'idle' && selectedList.length > 0 && (
                        <button onClick={() => onStartRun(selectedList)}
                            style={{ flexShrink: 0, width: '100%', padding: '7px', background: 'transparent', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 8, cursor: 'pointer', fontSize: 10, color: '#334155', fontWeight: 600 }}>
                            ▶ Start Without Reliability Check
                        </button>
                    )}
                </>
            )}
        </div>
    );
};

export default ModelSelectionPanel;
