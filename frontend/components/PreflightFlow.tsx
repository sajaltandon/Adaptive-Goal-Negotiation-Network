import React, { useState, useEffect } from 'react';

interface RankedModel {
    model: string;
    score: number;
    selected: boolean;
}

interface ReliabilityResult {
    model: string;
    status: 'checking' | 'ok' | 'fail';
    latency?: number;
    error?: string;
}

interface Props {
    discoveredModels: string[];
    onDone: (selectedModels: string[]) => void;
    apiUrl: string;
    lmUrl: string;
    prompt?: string;
}

const PreflightFlow: React.FC<Props> = ({ discoveredModels, onDone, apiUrl, lmUrl, prompt }) => {
    const [mode, setMode] = useState<'auto' | 'manual'>('auto');
    const [step, setStep] = useState<'select' | 'reliability' | 'ready'>('select');

    // Auto selection state
    const [autoLoading, setAutoLoading] = useState(false);
    const [autoRanked, setAutoRanked] = useState<RankedModel[]>([]);
    const [autoError, setAutoError] = useState('');
    const [autoTaskType, setAutoTaskType] = useState('');
    const [autoComplexity, setAutoComplexity] = useState('');

    // Manual selection state
    const [manualChecked, setManualChecked] = useState<Set<string>>(
        new Set(discoveredModels.slice(0, 3))
    );

    // Reliability state
    const [reliResults, setReliResults] = useState<ReliabilityResult[]>([]);

    // ── Trigger real auto-ranking when mode = auto ────────────────────────────
    useEffect(() => {
        if (mode !== 'auto' || step !== 'select') return;
        setAutoLoading(true);
        setAutoRanked([]);
        setAutoError('');

        fetch(`${apiUrl}/rank-models`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ base_url: lmUrl, prompt: prompt || '', max_agents: 4 }),
        })
            .then(res => res.json())
            .then(data => {
                setAutoRanked(data.ranked_models || []);
                setAutoTaskType(data.task_type || '');
                setAutoComplexity(data.complexity || '');
                setAutoLoading(false);
            })
            .catch(err => {
                setAutoError('Ranking failed — LM Studio may be busy. Switch to Manual.');
                setAutoLoading(false);
            });
    }, [mode, step]);

    // ── Run real reliability probe per model ──────────────────────────────────
    const runReliability = (models: string[]) => {
        setStep('reliability');
        const initial: ReliabilityResult[] = models.map(m => ({ model: m, status: 'checking' }));
        setReliResults(initial);

        // Probe models sequentially to avoid hammering LM Studio
        const probe = async (index: number) => {
            if (index >= models.length) {
                setTimeout(() => setStep('ready'), 400);
                return;
            }
            const model = models[index];
            try {
                const res = await fetch(`${apiUrl}/probe-model`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ base_url: lmUrl, model }),
                });
                const data = await res.json();
                setReliResults(prev => prev.map(r =>
                    r.model === model
                        ? { ...r, status: data.ok ? 'ok' : 'fail', latency: data.latency_ms, error: data.error }
                        : r
                ));
            } catch {
                setReliResults(prev => prev.map(r =>
                    r.model === model ? { ...r, status: 'fail', error: 'Network error' } : r
                ));
            }
            probe(index + 1);
        };
        probe(0);
    };

    const handleProceed = () => {
        const selected = mode === 'auto'
            ? autoRanked.filter(r => r.selected).map(r => r.model)
            : Array.from(manualChecked);
        if (selected.length === 0) return;
        runReliability(selected);
    };

    const handleLaunch = () => {
        const selected = mode === 'auto'
            ? autoRanked.filter(r => r.selected).map(r => r.model)
            : Array.from(manualChecked);
        onDone(selected);
    };

    const toggleManual = (model: string) => {
        setManualChecked(prev => {
            const next = new Set(prev);
            if (next.has(model)) { next.delete(model); } else { next.add(model); }
            return next;
        });
    };

    const autoSelected = autoRanked.filter(r => r.selected).map(r => r.model);
    const canProceed = step === 'select' && (
        (mode === 'auto' && !autoLoading && autoRanked.length > 0) ||
        (mode === 'manual' && manualChecked.size > 0)
    );

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

            {/* Mode Toggle */}
            <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12, padding: '16px 20px' }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                    Model Selection Mode
                </div>
                <div style={{ display: 'flex', gap: 0, background: 'rgba(0,0,0,0.3)', borderRadius: 8, padding: 3, width: 'fit-content' }}>
                    {(['auto', 'manual'] as const).map(m => (
                        <button key={m} onClick={() => { setMode(m); setStep('select'); setReliResults([]); }}
                            style={{
                                padding: '7px 20px', borderRadius: 6, border: 'none', cursor: 'pointer',
                                fontSize: 12, fontWeight: 700, transition: 'all 0.2s',
                                background: mode === m ? 'linear-gradient(135deg,#22d3ee22,#818cf833)' : 'transparent',
                                color: mode === m ? '#22d3ee' : '#475569',
                                boxShadow: mode === m ? 'inset 0 0 0 1px #22d3ee33' : 'none',
                            }}>
                            {m === 'auto' ? '⚡ Auto' : '☑ Manual'}
                        </button>
                    ))}
                </div>

                {/* ── Auto mode ── */}
                {mode === 'auto' && step === 'select' && (
                    <div style={{ marginTop: 14 }}>
                        {autoLoading && (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#64748b', fontSize: 12 }}>
                                    <span style={{ display: 'inline-block', width: 12, height: 12, borderRadius: '50%', border: '2px solid #22d3ee', borderTopColor: 'transparent', animation: 'spin 0.8s linear infinite', flexShrink: 0 }} />
                                    Running model selection... (analyze_task + select_models)
                                </div>
                                {discoveredModels.map(m => (
                                    <div key={m} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                        <div style={{ flex: 1, height: 3, background: 'rgba(255,255,255,0.05)', borderRadius: 2, overflow: 'hidden' }}>
                                            <div style={{ height: '100%', width: '60%', background: 'linear-gradient(90deg,#22d3ee33,#818cf866)', borderRadius: 2, animation: 'shimmer 1.4s ease-in-out infinite' }} />
                                        </div>
                                        <span style={{ fontSize: 10, color: '#334155', fontFamily: 'monospace', minWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m}</span>
                                    </div>
                                ))}
                            </div>
                        )}

                        {autoError && (
                            <div style={{ fontSize: 11, color: '#f87171', padding: '8px 12px', background: 'rgba(248,113,113,0.08)', borderRadius: 8, border: '1px solid #f8717133' }}>
                                ⚠ {autoError}
                            </div>
                        )}

                        {!autoLoading && autoRanked.length > 0 && (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                                {(autoTaskType || autoComplexity) && (
                                    <div style={{ display: 'flex', gap: 6, marginBottom: 4, flexWrap: 'wrap' }}>
                                        {autoTaskType && <span style={{ fontSize: 9, background: 'rgba(99,102,241,0.15)', color: '#818cf8', border: '1px solid #818cf833', borderRadius: 4, padding: '2px 8px', fontWeight: 700 }}>type: {autoTaskType}</span>}
                                        {autoComplexity && <span style={{ fontSize: 9, background: 'rgba(251,146,60,0.1)', color: '#fb923c', border: '1px solid #fb923c33', borderRadius: 4, padding: '2px 8px', fontWeight: 700 }}>complexity: {autoComplexity}</span>}
                                        <span style={{ fontSize: 9, color: '#475569' }}>— AGNN selected the top {autoSelected.length} models</span>
                                    </div>
                                )}
                                {autoRanked.map((r, i) => (
                                    <div key={r.model} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '7px 10px', background: r.selected ? 'rgba(34,211,238,0.06)' : 'rgba(255,255,255,0.02)', border: `1px solid ${r.selected ? '#22d3ee33' : 'rgba(255,255,255,0.05)'}`, borderRadius: 7 }}>
                                        <span style={{ fontSize: 10, color: r.selected ? '#22d3ee' : '#334155', fontWeight: 700, minWidth: 18 }}>#{i + 1}</span>
                                        <span style={{ flex: 1, fontSize: 11, color: r.selected ? '#e2e8f0' : '#475569', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.model}</span>
                                        <span style={{ fontSize: 10, color: r.selected ? '#22d3ee' : '#334155', fontWeight: 700, minWidth: 36 }}>{r.score.toFixed(2)}</span>
                                        {r.selected && <span style={{ fontSize: 12, color: '#22d3ee' }}>✓</span>}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}

                {/* ── Manual mode ── */}
                {mode === 'manual' && step === 'select' && (
                    <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 6 }}>
                        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 2 }}>Select models to include ({manualChecked.size} selected)</div>
                        {discoveredModels.map(m => {
                            const checked = manualChecked.has(m);
                            return (
                                <label key={m} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px', background: checked ? 'rgba(34,211,238,0.06)' : 'rgba(255,255,255,0.02)', border: `1px solid ${checked ? '#22d3ee44' : 'rgba(255,255,255,0.06)'}`, borderRadius: 8, cursor: 'pointer', transition: 'all 0.2s' }}>
                                    <div style={{ width: 18, height: 18, borderRadius: 5, border: `2px solid ${checked ? '#22d3ee' : '#334155'}`, background: checked ? '#22d3ee' : 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, transition: 'all 0.2s' }}>
                                        {checked && <span style={{ fontSize: 11, color: '#030712', fontWeight: 900 }}>✓</span>}
                                    </div>
                                    <input type="checkbox" checked={checked} onChange={() => toggleManual(m)} style={{ display: 'none' }} />
                                    <span style={{ flex: 1, fontSize: 12, color: checked ? '#e2e8f0' : '#64748b', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m}</span>
                                </label>
                            );
                        })}
                    </div>
                )}

                {/* Proceed button */}
                {step === 'select' && canProceed && (
                    <button onClick={handleProceed}
                        style={{ marginTop: 14, width: '100%', padding: '9px', background: 'rgba(34,211,238,0.1)', border: '1px solid #22d3ee44', borderRadius: 8, cursor: 'pointer', fontSize: 12, color: '#22d3ee', fontWeight: 700 }}>
                        → Run Model Reliability Check
                    </button>
                )}
            </div>

            {/* ── Reliability Check ── */}
            {(step === 'reliability' || step === 'ready') && reliResults.length > 0 && (
                <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12, padding: '16px 20px' }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                        Model Reliability Check
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {reliResults.map(r => (
                            <div key={r.model} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px', background: r.status === 'ok' ? 'rgba(52,211,153,0.05)' : r.status === 'fail' ? 'rgba(248,113,113,0.05)' : 'rgba(255,255,255,0.02)', border: `1px solid ${r.status === 'ok' ? '#34d39933' : r.status === 'fail' ? '#f8717133' : 'rgba(255,255,255,0.06)'}`, borderRadius: 8, transition: 'all 0.4s' }}>
                                {r.status === 'checking' && <span style={{ width: 16, height: 16, borderRadius: '50%', border: '2px solid #22d3ee', borderTopColor: 'transparent', animation: 'spin 0.8s linear infinite', display: 'inline-block', flexShrink: 0 }} />}
                                {r.status === 'ok' && <span style={{ fontSize: 14, flexShrink: 0 }}>✅</span>}
                                {r.status === 'fail' && <span style={{ fontSize: 14, flexShrink: 0 }}>❌</span>}
                                <span style={{ flex: 1, fontSize: 11, color: '#94a3b8', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.model}</span>
                                {r.status === 'checking' && <span style={{ fontSize: 10, color: '#475569' }}>probing...</span>}
                                {r.status === 'ok' && <span style={{ fontSize: 10, color: '#34d399', fontWeight: 700 }}>{r.latency}ms</span>}
                                {r.status === 'fail' && <span style={{ fontSize: 10, color: '#f87171' }}>{r.error || 'unreachable'}</span>}
                            </div>
                        ))}
                    </div>

                    {step === 'ready' && (
                        <>
                            <div style={{ marginTop: 12, padding: '8px 12px', background: 'rgba(52,211,153,0.08)', border: '1px solid #34d39933', borderRadius: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
                                <span>✓</span>
                                <span style={{ fontSize: 11, color: '#34d399', fontWeight: 600 }}>
                                    {reliResults.filter(r => r.status === 'ok').length}/{reliResults.length} models verified — ready to start negotiation
                                </span>
                            </div>
                            <button onClick={handleLaunch}
                                style={{ marginTop: 12, width: '100%', padding: '13px', background: 'linear-gradient(135deg,#22d3ee22,#818cf833)', border: '1px solid #22d3ee55', borderRadius: 10, cursor: 'pointer', fontSize: 14, color: '#22d3ee', fontWeight: 800, letterSpacing: '-0.3px', boxShadow: '0 0 20px #22d3ee11' }}>
                                ▶ Launch AGNN Run
                            </button>
                        </>
                    )}
                </div>
            )}
        </div>
    );
};

export default PreflightFlow;
