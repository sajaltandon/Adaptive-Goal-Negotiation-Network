import React from 'react';
import { TeamFormationData, NegotiationMessage } from '../types';

interface Props {
    data: TeamFormationData | null;
    taskBadge?: { type: string; complexity: string; budget: string };
}

const agentColors: Record<string, string> = {
    AgentA: '#22d3ee',
    AgentB: '#a78bfa',
    AgentC: '#34d399',
    AgentD: '#fb923c',
    AgentE: '#f472b6',
};
const getColor = (id: string) => agentColors[id] || '#94a3b8';

const StatusChip: React.FC<{ status: string }> = ({ status }) => {
    const map: Record<string, { bg: string; text: string; label: string }> = {
        negotiating: { bg: 'rgba(251,191,36,0.15)', text: '#fbbf24', label: 'negotiating' },
        active: { bg: 'rgba(52,211,153,0.15)', text: '#34d399', label: 'active' },
        idle: { bg: 'rgba(148,163,184,0.1)', text: '#94a3b8', label: 'idle' },
        'hot-swapping': { bg: 'rgba(251,146,60,0.15)', text: '#fb923c', label: 'hot-swapping' },
    };
    const s = map[status] || map.idle;
    return (
        <span style={{ background: s.bg, color: s.text, border: `1px solid ${s.text}33`, borderRadius: 4, fontSize: 9, fontWeight: 700, padding: '2px 6px', textTransform: 'uppercase', letterSpacing: 1 }}>
            {s.label}
        </span>
    );
};

const TISBar: React.FC<{ value: number }> = ({ value }) => (
    <div style={{ flex: 1, height: 4, background: 'rgba(255,255,255,0.07)', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ width: `${Math.round(value * 100)}%`, height: '100%', background: 'linear-gradient(90deg,#22d3ee,#818cf8)', borderRadius: 2, transition: 'width 0.5s' }} />
    </div>
);

const TeamFormationPanel: React.FC<Props> = ({ data, taskBadge }) => {
    const strength = data?.consensusStrength ?? 0;
    const members = data?.members ?? [];
    const negotiation: NegotiationMessage[] = data?.negotiation ?? [];

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: 10 }}>
            {/* Task badge */}
            {taskBadge && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                    <span style={{ background: 'rgba(99,102,241,0.15)', color: '#818cf8', border: '1px solid #818cf833', borderRadius: 4, fontSize: 10, padding: '2px 8px', fontWeight: 600 }}>
                        {taskBadge.type}
                    </span>
                    <span style={{ background: 'rgba(244,63,94,0.12)', color: '#fb7185', border: '1px solid #fb718533', borderRadius: 4, fontSize: 10, padding: '2px 8px', fontWeight: 600 }}>
                        {taskBadge.complexity}
                    </span>
                    <span style={{ background: 'rgba(52,211,153,0.1)', color: '#34d399', border: '1px solid #34d39933', borderRadius: 4, fontSize: 10, padding: '2px 8px', fontWeight: 600 }}>
                        {taskBadge.budget}
                    </span>
                </div>
            )}

            {/* Agent cards */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {members.length > 0 ? (
                    // Final members after negotiation
                    members.map((m) => {
                        const id = m.agent_id;
                        const role = m.role?.name || 'Unassigned';
                        const secondaries = (m.secondary_roles || []).map(r => r?.name || r).filter(Boolean);
                        return (
                            <div key={id} style={{ background: 'rgba(255,255,255,0.03)', border: `1px solid ${getColor(id)}22`, borderRadius: 8, padding: '8px 10px', display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                                <div style={{ width: 30, height: 30, borderRadius: '50%', background: `${getColor(id)}18`, border: `2px solid ${getColor(id)}66`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 700, color: getColor(id), flexShrink: 0 }}>{id.replace('Agent', '')}</div>
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{ fontSize: 11, fontWeight: 700, color: '#e2e8f0', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{role}</div>
                                    <div style={{ fontSize: 9, color: '#64748b', marginTop: 1, fontFamily: 'monospace' }}>{m.model}</div>
                                    {secondaries.length > 0 && (
                                        <div style={{ marginTop: 3, display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                                            {secondaries.map(s => (
                                                <span key={s} style={{ fontSize: 8, background: 'rgba(167,139,250,0.1)', color: '#a78bfa', border: '1px solid #a78bfa33', borderRadius: 3, padding: '1px 5px' }}>+{s}</span>
                                            ))}
                                        </div>
                                    )}
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 5 }}>
                                        <span style={{ fontSize: 8, color: '#475569' }}>TIS</span>
                                        <TISBar value={m.confidence || 0.5} />
                                    </div>
                                </div>
                                <StatusChip status="active" />
                            </div>
                        );
                    })
                ) : (
                    // During role generation / negotiation
                    (data?.roles || []).length > 0 || negotiation.length > 0 ? (
                        Array.from(new Set(negotiation.map(n => n.agent))).map(id => (
                            <div key={id} style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 8, padding: '8px 10px', display: 'flex', alignItems: 'center', gap: 10 }}>
                                <div style={{ width: 28, height: 28, borderRadius: '50%', background: `${getColor(id)}22`, border: `2px solid ${getColor(id)}55`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, fontWeight: 700, color: getColor(id) }}>{id.replace('Agent', '')}</div>
                                <div style={{ flex: 1 }}>
                                    <div style={{ fontSize: 11, fontWeight: 600, color: '#cbd5e1' }}>{id}</div>
                                    <div style={{ fontSize: 9, color: '#fbbf24' }}>Negotiating role...</div>
                                </div>
                                <StatusChip status="negotiating" />
                            </div>
                        ))
                    ) : (
                        // Very start (roles are generating)
                        <div style={{ color: '#64748b', fontSize: 11, textAlign: 'center', padding: '10px 0' }}>Creating tasks and roles for this team...</div>
                    )
                )}
            </div>

            {/* Consensus meter */}
            <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 8, padding: '8px 10px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                    <span style={{ fontSize: 10, color: '#94a3b8', fontWeight: 600 }}>Consensus Strength</span>
                    <span style={{ fontSize: 10, color: '#22d3ee', fontWeight: 700 }}>{strength.toFixed(2)}</span>
                </div>
                <div style={{ height: 5, background: 'rgba(255,255,255,0.07)', borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{ width: `${strength * 100}%`, height: '100%', background: 'linear-gradient(90deg, #22d3ee, #818cf8)', borderRadius: 3, transition: 'width 1s' }} />
                </div>
            </div>

            {/* Role negotiation chat */}
            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 5, paddingRight: '4px' }}>
                {(() => {
                    const primary = negotiation.filter(m => m.phase === 'primary' || !m.phase);
                    const secondary = negotiation.filter(m => m.phase === 'secondary');

                    return (
                        <>
                            {primary.length > 0 && (
                                <div style={{ fontSize: 9, fontWeight: 800, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5, marginTop: 4, marginBottom: 2 }}>
                                    Negotiating Primary Roles
                                </div>
                            )}
                            {primary.map((msg, i) => (
                                <div key={`pri-${i}`} style={{ background: 'rgba(255,255,255,0.025)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: 6, padding: '6px 8px' }}>
                                    <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 2 }}>
                                        <span style={{ fontSize: 9, fontWeight: 700, color: getColor(msg.agent) }}>{msg.agent}</span>
                                        {(msg.proposed_role || msg.final_role) && (
                                            <span style={{ fontSize: 8, color: '#64748b' }}>→ {msg.proposed_role || msg.final_role}</span>
                                        )}
                                        <span style={{ marginLeft: 'auto', fontSize: 8, color: '#334155' }}>Turn {msg.turn}</span>
                                    </div>
                                    <div style={{ fontSize: 10, color: '#94a3b8', lineHeight: 1.4 }}>
                                        {msg.content}
                                    </div>
                                </div>
                            ))}

                            {secondary.length > 0 && (
                                <div style={{ fontSize: 9, fontWeight: 800, color: '#a78bfa', textTransform: 'uppercase', letterSpacing: 0.5, marginTop: 12, marginBottom: 2, borderTop: '1px dashed rgba(167,139,250,0.2)', paddingTop: 8 }}>
                                    Negotiating Uncovered Roles
                                </div>
                            )}
                            {secondary.map((msg, i) => (
                                <div key={`sec-${i}`} style={{ background: 'rgba(167,139,250,0.03)', border: '1px solid rgba(167,139,250,0.1)', borderRadius: 6, padding: '6px 8px' }}>
                                    <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 2 }}>
                                        <span style={{ fontSize: 9, fontWeight: 700, color: getColor(msg.agent) }}>{msg.agent}</span>
                                        {(msg.proposed_role || msg.final_role) && (
                                            <span style={{ fontSize: 8, color: '#a78bfa' }}>→ {msg.proposed_role || msg.final_role}</span>
                                        )}
                                        <span style={{ marginLeft: 'auto', fontSize: 8, color: '#334155' }}>Turn {msg.turn}</span>
                                    </div>
                                    <div style={{ fontSize: 10, color: '#94a3b8', lineHeight: 1.4 }}>
                                        {msg.content}
                                    </div>
                                </div>
                            ))}

                            {negotiation.length === 0 && (
                                <div style={{ color: '#334155', fontSize: 11, textAlign: 'center', marginTop: 10 }}>Waiting for negotiation...</div>
                            )}
                        </>
                    );
                })()}
            </div>
        </div>
    );
};

export default TeamFormationPanel;
