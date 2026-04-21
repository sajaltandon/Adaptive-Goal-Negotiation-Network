import React from 'react';
import { QualityScores } from '../types';

interface Props {
    synthStatus: 'pending' | 'running' | 'done';
    scoringStatus: 'pending' | 'running' | 'done';
    outputReady: boolean;
    scores: QualityScores;
    onViewDoc: () => void;
    onDownload: () => void;
}

const DIMS: { key: keyof QualityScores; label: string; color: string }[] = [
    { key: 'coverage', label: 'Coverage', color: '#22d3ee' },
    { key: 'depth', label: 'Depth', color: '#818cf8' },
    { key: 'coherence', label: 'Coherence', color: '#a78bfa' },
    { key: 'accuracy', label: 'Accuracy', color: '#34d399' },
    { key: 'actionability', label: 'Actionability', color: '#fb923c' },
    { key: 'formatting', label: 'Formatting', color: '#f472b6' },
];

const MiniGauge: React.FC<{ value: number; label: string; color: string }> = ({ value, label, color }) => {
    const r = 20;
    const circ = 2 * Math.PI * r;
    const dash = circ * Math.min(1, Math.max(0, value));
    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
            <div style={{ position: 'relative', width: 50, height: 50 }}>
                <svg width="50" height="50" style={{ transform: 'rotate(-90deg)' }}>
                    <circle cx="25" cy="25" r={r} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="3.5" />
                    <circle cx="25" cy="25" r={r} fill="none" stroke={color} strokeWidth="3.5"
                        strokeDasharray={`${dash} ${circ}`} strokeLinecap="round"
                        style={{ transition: 'stroke-dasharray 1s ease', filter: `drop-shadow(0 0 4px ${color}66)` }} />
                </svg>
                <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 700, color }}>
                    {Math.round(value * 100)}
                </div>
            </div>
            <span style={{ fontSize: 9, color: '#64748b', textAlign: 'center', fontWeight: 600 }}>{label}</span>
        </div>
    );
};

const Pill: React.FC<{ label: string; status: 'pending' | 'running' | 'done' }> = ({ label, status }) => {
    const colors = { pending: '#334155', running: '#22d3ee', done: '#34d399' } as const;
    const bg = { pending: 'rgba(51,65,85,0.3)', running: 'rgba(34,211,238,0.1)', done: 'rgba(52,211,153,0.1)' } as const;
    return (
        <div style={{ display: 'flex', alignItems: 'center', gap: 5, background: bg[status], border: `1px solid ${colors[status]}33`, borderRadius: 20, padding: '4px 12px' }}>
            {status === 'running' && (
                <span style={{ display: 'inline-block', width: 7, height: 7, borderRadius: '50%', border: `1.5px solid ${colors[status]}`, borderTopColor: 'transparent', animation: 'spin 1s linear infinite' }} />
            )}
            {status === 'done' && <span style={{ fontSize: 10, color: colors[status] }}>✓</span>}
            <span style={{ fontSize: 10, fontWeight: 600, color: colors[status] }}>{label}</span>
        </div>
    );
};

const BottomBar: React.FC<Props> = ({ synthStatus, scoringStatus, outputReady, scores, onViewDoc, onDownload }) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '8px 16px', background: 'rgba(2,6,23,0.8)', borderTop: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(12px)', flexWrap: 'wrap' }}>
        {/* Flow pills */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
            <Pill label="Synthesis" status={synthStatus} />
            <svg width="16" height="10"><path d="M0 5 L12 5 M9 2 L12 5 L9 8" stroke="#1e293b" strokeWidth="1.5" fill="none" /></svg>
            <Pill label="LLM Scoring" status={scoringStatus} />
            {outputReady && (
                <>
                    <svg width="16" height="10"><path d="M0 5 L12 5 M9 2 L12 5 L9 8" stroke="#1e293b" strokeWidth="1.5" fill="none" /></svg>
                    <Pill label="Output Ready" status="done" />
                </>
            )}
        </div>

        {/* Quality gauges */}
        <div style={{ flex: 1, display: 'flex', justifyContent: 'center', gap: 14, alignItems: 'flex-end', flexWrap: 'wrap' }}>
            {DIMS.map(d => (
                <MiniGauge key={d.key} value={scores[d.key]} label={d.label} color={d.color} />
            ))}
        </div>

        {/* Buttons */}
        <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
            <button onClick={onDownload} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 14px', background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 7, cursor: 'pointer', fontSize: 11, color: '#cbd5e1', fontWeight: 600, transition: 'all 0.2s' }} onMouseOver={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.1)')} onMouseOut={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.06)')}>
                ↓ Download .md
            </button>
            <button onClick={onViewDoc} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 14px', background: 'linear-gradient(135deg, #22d3ee22, #818cf822)', border: '1px solid #22d3ee44', borderRadius: 7, cursor: 'pointer', fontSize: 11, color: '#22d3ee', fontWeight: 700, transition: 'all 0.2s', boxShadow: '0 0 10px #22d3ee11' }} onMouseOver={e => (e.currentTarget.style.boxShadow = '0 0 16px #22d3ee33')} onMouseOut={e => (e.currentTarget.style.boxShadow = '0 0 10px #22d3ee11')}>
                📄 View Full Document
            </button>
        </div>
    </div>
);

export default BottomBar;
