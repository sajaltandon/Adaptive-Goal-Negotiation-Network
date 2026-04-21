import React from 'react';
import { AgentWorkspaceState, WorkspaceStatus } from '../types';

interface Props {
    workspaces: AgentWorkspaceState[];
}

const agentColors: Record<string, string> = {
    AgentA: '#22d3ee', AgentB: '#a78bfa', AgentC: '#34d399',
    AgentD: '#fb923c', AgentE: '#f472b6',
};
const getColor = (id: string) => agentColors[id] || '#64748b';

const StatusFooter: React.FC<{ status: WorkspaceStatus }> = ({ status }) => {
    const map: Record<WorkspaceStatus, { bg: string; text: string; label: string; icon: string }> = {
        'pending': { bg: 'rgba(71,85,105,0.3)', text: '#64748b', label: 'PENDING', icon: '⏳' },
        'generating': { bg: 'rgba(34,211,238,0.1)', text: '#22d3ee', label: 'GENERATING...', icon: '◌' },
        'accepted': { bg: 'rgba(52,211,153,0.1)', text: '#34d399', label: 'ACCEPTED ✓', icon: '✓' },
        'rejected': { bg: 'rgba(248,113,113,0.1)', text: '#f87171', label: 'REJECTED ✗', icon: '✗' },
        'hot-swapping': { bg: 'rgba(251,146,60,0.12)', text: '#fb923c', label: 'HOT-SWAP ⚡', icon: '⚡' },
        'done': { bg: 'rgba(52,211,153,0.08)', text: '#34d399', label: 'DONE ✓', icon: '✓' },
    };
    const s = map[status] || map.pending;
    return (
        <div style={{ background: s.bg, borderTop: `1px solid ${s.text}22`, padding: '5px 10px', display: 'flex', alignItems: 'center', gap: 5, borderBottomLeftRadius: 10, borderBottomRightRadius: 10 }}>
            {status === 'generating' && (
                <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', border: `2px solid ${s.text}`, borderTopColor: 'transparent', animation: 'spin 1s linear infinite' }} />
            )}
            <span style={{ fontSize: 10, fontWeight: 700, color: s.text, letterSpacing: 0.5 }}>{s.label}</span>
        </div>
    );
};

const TISRing: React.FC<{ value: number; color: string }> = ({ value, color }) => {
    const r = 16;
    const circ = 2 * Math.PI * r;
    const dash = circ * Math.min(1, Math.max(0, value));
    return (
        <div style={{ position: 'relative', width: 44, height: 44 }}>
            <svg width="44" height="44" style={{ transform: 'rotate(-90deg)' }}>
                <circle cx="22" cy="22" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="3" />
                <circle cx="22" cy="22" r={r} fill="none" stroke={color} strokeWidth="3"
                    strokeDasharray={`${dash} ${circ}`} strokeLinecap="round" style={{ transition: 'stroke-dasharray 0.6s ease' }} />
            </svg>
            <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 9, fontWeight: 700, color }}>
                {Math.round(value * 100)}
            </div>
        </div>
    );
};

const WorkspaceCard: React.FC<{ ws: AgentWorkspaceState }> = ({ ws }) => {
    const color = getColor(ws.agent_id);
    return (
        <div style={{ background: 'rgba(255,255,255,0.025)', border: `1px solid ${color}22`, borderRadius: 10, display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative' }}>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ width: 24, height: 24, borderRadius: '50%', background: `${color}18`, border: `2px solid ${color}55`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, fontWeight: 700, color, flexShrink: 0 }}>
                    {ws.agent_id.replace('Agent', '')}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: '#e2e8f0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{ws.primary_role}</div>
                    <div style={{ fontSize: 8, color: '#475569', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{ws.model}</div>
                </div>
            </div>

            {/* Body */}
            <div style={{ padding: '8px 10px', flex: 1 }}>
                {/* Subgoal */}
                <div style={{ fontSize: 9, color: '#64748b', marginBottom: 2 }}>Current subgoal</div>
                <div style={{ fontSize: 10, color: '#94a3b8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginBottom: 8 }}>
                    {ws.current_subgoal || 'Waiting...'}
                </div>

                {/* Turn counter */}
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <span style={{ fontSize: 9, color: '#475569' }}>Turn <span style={{ color: '#94a3b8', fontWeight: 700 }}>{ws.turn_count}/{ws.max_turns}</span></span>
                    <span style={{ fontSize: 9, color: '#475569' }}>{ws.elapsed_s}s</span>
                </div>

                {/* TIS + rejections */}
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                    <div>
                        <div style={{ fontSize: 8, color: '#475569', marginBottom: 3 }}>TIS Score</div>
                        <TISRing value={ws.tis_score} color={color} />
                    </div>
                    {ws.metrics && (
                        <div style={{ flex: 1, display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '2px 8px' }}>
                            {['SD', 'RC', 'IS', 'EIC', 'St'].map(key => (
                                <div key={key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span style={{ fontSize: 8, color: '#475569' }}>{key}</span>
                                    <span style={{ fontSize: 8, color: (ws.metrics![key] || 0) > 0.2 ? '#10b981' : '#f59e0b', fontWeight: 600 }}>
                                        {Number(ws.metrics![key] || 0).toFixed(2)}
                                    </span>
                                </div>
                            ))}
                        </div>
                    )}
                    {ws.rejection_count > 0 && (
                        <div style={{ background: 'rgba(248,113,113,0.15)', border: '1px solid #f8717133', borderRadius: 6, padding: '4px 6px', textAlign: 'center' }}>
                            <div style={{ fontSize: 12, fontWeight: 700, color: '#f87171' }}>{ws.rejection_count}</div>
                            <div style={{ fontSize: 7, color: '#f87171' }}>rejects</div>
                        </div>
                    )}
                </div>

                {/* Last message preview */}
                <div style={{ marginTop: 8, background: 'rgba(0,0,0,0.2)', borderRadius: 5, padding: '5px 7px' }}>
                    <div style={{ fontSize: 8, color: '#334155', marginBottom: 2, textTransform: 'uppercase', letterSpacing: 0.5 }}>Last message</div>
                    <div style={{ fontSize: 9, color: '#64748b', lineHeight: 1.5, overflow: 'hidden', display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical' }}>
                        {ws.last_message || '—'}
                    </div>
                </div>
            </div>

            <StatusFooter status={ws.status} />
        </div>
    );
};

const WorkspacesPanel: React.FC<Props> = ({ workspaces }) => {
    const display = workspaces.length > 0
        ? workspaces
        : ['AgentA', 'AgentB', 'AgentC', 'AgentD'].map(id => ({
            agent_id: id, model: '—', primary_role: id, secondary_roles: [],
            turn_count: 0, max_turns: 12, tis_score: 0, rejection_count: 0,
            last_message: '', status: 'pending' as const, elapsed_s: 0,
        }));

    return (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gridTemplateRows: 'auto auto', gap: 8, height: '100%', overflowY: 'auto' }}>
            {display.slice(0, 4).map(ws => (
                <WorkspaceCard key={ws.agent_id} ws={ws} />
            ))}
        </div>
    );
};

export default WorkspacesPanel;
