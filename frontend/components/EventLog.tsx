import React, { useState, useRef, useEffect } from 'react';
import { ChartEvent } from '../types';

interface Props {
    events: ChartEvent[];
    sessionId: string;
    elapsed: number;
}

const EventLog: React.FC<Props> = ({ events, sessionId, elapsed }) => {
    const [filter, setFilter] = useState<'all' | 'hotswap' | 'phase' | 'error'>('all');
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [events.length]);

    const kindColor: Record<string, string> = {
        phase: '#818cf8', role: '#a78bfa', plan: '#22d3ee',
        system: '#475569', error: '#f87171', hotswap: '#fb923c',
    };
    const filtered = filter === 'all' ? events : events.filter(e => e.kind === filter);
    const fmtElapsed = `${String(Math.floor(elapsed / 3600)).padStart(2, '0')}:${String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0')}:${String(elapsed % 60).padStart(2, '0')}`;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            {/* Session header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, paddingBottom: 8, borderBottom: '1px solid rgba(255,255,255,0.06)', marginBottom: 8, flexWrap: 'wrap' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    <div style={{ width: 7, height: 7, borderRadius: '50%', background: '#34d399', boxShadow: '0 0 6px #34d399' }} />
                    <span style={{ fontSize: 9, color: '#34d399', fontWeight: 600 }}>LIVE</span>
                </div>
                <span style={{ fontSize: 9, color: '#334155', fontFamily: 'monospace' }}>{sessionId}</span>
                <span style={{ marginLeft: 'auto', fontSize: 10, color: '#475569', fontFamily: 'monospace' }}>{fmtElapsed}</span>
            </div>

            {/* Filter chips */}
            <div style={{ display: 'flex', gap: 5, marginBottom: 8, flexWrap: 'wrap' }}>
                {(['all', 'hotswap', 'phase', 'error'] as const).map(f => (
                    <button key={f} onClick={() => setFilter(f)} style={{ fontSize: 9, padding: '2px 8px', borderRadius: 4, border: `1px solid ${filter === f ? '#22d3ee55' : 'rgba(255,255,255,0.08)'}`, background: filter === f ? 'rgba(34,211,238,0.1)' : 'transparent', color: filter === f ? '#22d3ee' : '#475569', cursor: 'pointer', fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.5 }}>
                        {f}
                    </button>
                ))}
            </div>

            {/* Events */}
            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 3 }}>
                {filtered.slice(-60).map((e, i) => (
                    <div key={i} style={{ display: 'flex', gap: 7, alignItems: 'flex-start', padding: '3px 0', borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                        <span style={{ fontSize: 9, color: '#334155', fontFamily: 'monospace', flexShrink: 0, marginTop: 1 }}>{e.timestamp}</span>
                        <span style={{ width: 3, flexShrink: 0, borderRadius: 2, alignSelf: 'stretch', background: kindColor[e.kind] || '#475569', opacity: 0.7 }} />
                        <span style={{ fontSize: 10, color: '#64748b', lineHeight: 1.4 }}>{e.label}</span>
                    </div>
                ))}
                {filtered.length === 0 && (
                    <div style={{ color: '#334155', fontSize: 11, textAlign: 'center', marginTop: 10 }}>No events yet...</div>
                )}
                <div ref={bottomRef} />
            </div>
        </div>
    );
};

export default EventLog;
