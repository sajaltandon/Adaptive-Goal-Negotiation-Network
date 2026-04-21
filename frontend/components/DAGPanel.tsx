import React, { useEffect } from 'react';
import { SubgoalInfo, SubgoalStatus } from '../types';
import { ReactFlow, Background, Handle, Position, Node, Edge, useNodesState, useEdgesState, MarkerType } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import dagre from 'dagre';

interface Props {
    subgoals: SubgoalInfo[];
    semaphoreSlots: number;
    semaphoreActive: number;
}

const agentColors: Record<string, string> = {
    AgentA: '#22d3ee', AgentB: '#a78bfa', AgentC: '#34d399',
    AgentD: '#fb923c', AgentE: '#f472b6',
};
const getColor = (id?: string) => (id && agentColors[id]) ? agentColors[id] : '#64748b';

const StatusDot: React.FC<{ status: SubgoalStatus }> = ({ status }) => {
    const col = { completed: '#34d399', running: '#22d3ee', failed: '#f87171', pending: '#475569' }[status];
    return (
        <span style={{ display: 'inline-block', width: 7, height: 7, borderRadius: '50%', background: col, marginRight: 5, flexShrink: 0, boxShadow: status === 'running' ? `0 0 6px ${col}` : 'none' }} />
    );
};

const SubgoalNode = ({ data }: { data: SubgoalInfo }) => {
    const isRunning = data.status === 'running';

    let bg = 'rgba(255,255,255,0.02)';
    let border = '1px dashed rgba(255,255,255,0.1)';
    let color = '#475569';
    let boxShadow = 'none';

    if (data.status === 'completed') {
        bg = 'rgba(52,211,153,0.1)'; border = '1px solid #34d39944'; color = '#34d399';
    } else if (data.status === 'failed') {
        bg = 'rgba(248,113,113,0.1)'; border = '1px solid #f8717144'; color = '#f87171';
    } else if (data.status === 'running') {
        bg = 'rgba(34,211,238,0.07)'; border = '1px solid #22d3ee55'; color = '#e2e8f0';
        boxShadow = '0 0 12px #22d3ee22';
    }

    return (
        <div style={{
            background: bg, border, color, boxShadow,
            borderRadius: 8, padding: '8px 10px', fontSize: 11,
            width: 250, position: 'relative', overflow: 'hidden',
            transition: 'all 0.3s ease',
            animation: data.status === 'completed' ? 'pop 0.4s ease-out' : (isRunning ? 'pulseBorder 2s infinite' : 'none')
        }}>
            <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
            <Handle type="source" position={Position.Right} style={{ opacity: 0 }} />

            {isRunning && (
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: 'linear-gradient(90deg, transparent, #22d3ee, transparent)', animation: 'shimmer 2s infinite' }} />
            )}

            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
                <div style={{ width: 22, height: 22, borderRadius: '50%', background: `${getColor(data.assigned_agent)}18`, border: `2px solid ${getColor(data.assigned_agent)}55`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 9, fontWeight: 700, color: getColor(data.assigned_agent), flexShrink: 0 }}>
                    {data.assigned_agent?.replace('Agent', '') || '?'}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <StatusDot status={data.status} />
                        <span style={{ fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{data.name}</span>
                    </div>
                    {data.assigned_role && (
                        <div style={{ fontSize: 9, color: '#64748b', marginTop: 2 }}>{data.assigned_role}</div>
                    )}
                </div>
                {data.status === 'completed' && <span style={{ fontSize: 14 }}>✓</span>}
            </div>
        </div>
    );
};

const nodeTypes = { subgoal: SubgoalNode };

const getLayoutedElements = (nodes: Node[], edges: Edge[]) => {
    const dagreGraph = new dagre.graphlib.Graph();
    dagreGraph.setDefaultEdgeLabel(() => ({}));
    // Add some spacing so the edges are easy to see
    dagreGraph.setGraph({ rankdir: 'LR', nodesep: 40, ranksep: 80 });

    nodes.forEach((node) => {
        dagreGraph.setNode(node.id, { width: 250, height: 60 });
    });

    edges.forEach((edge) => {
        dagreGraph.setEdge(edge.source, edge.target);
    });

    dagre.layout(dagreGraph);

    const layoutedNodes = nodes.map((node) => {
        const nodeWithPosition = dagreGraph.node(node.id);
        return {
            ...node,
            targetPosition: Position.Left,
            sourcePosition: Position.Right,
            position: {
                x: nodeWithPosition.x - 250 / 2,
                y: nodeWithPosition.y - 60 / 2,
            },
        };
    });

    return { nodes: layoutedNodes, edges };
};

const DAGPanel: React.FC<Props> = ({ subgoals, semaphoreSlots, semaphoreActive }) => {
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);

    useEffect(() => {
        if (!subgoals || subgoals.length === 0) {
            setNodes([]);
            setEdges([]);
            return;
        }

        const initialNodes: Node[] = subgoals.map(sg => ({
            id: String(sg.id),
            type: 'subgoal',
            position: { x: 0, y: 0 },
            data: sg
        }));

        const initialEdges: Edge[] = [];
        subgoals.forEach(sg => {
            (sg.dependencies || []).forEach(dep => {
                const depNode = subgoals.find(n => String(n.id) === String(dep));
                if (depNode) {
                    const isRunningOrDone = sg.status !== 'pending' && sg.status !== 'failed';
                    initialEdges.push({
                        id: `e${dep}-${sg.id}`,
                        source: String(dep),
                        target: String(sg.id),
                        type: 'smoothstep',
                        animated: sg.status === 'running',
                        style: { stroke: isRunningOrDone ? '#22d3ee' : '#334155', strokeWidth: 1.5 },
                        markerEnd: {
                            type: MarkerType.ArrowClosed,
                            color: isRunningOrDone ? '#22d3ee' : '#334155',
                        },
                    });
                }
            });
        });

        const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(initialNodes, initialEdges);
        setNodes(layoutedNodes);
        setEdges(layoutedEdges);
    }, [subgoals, setNodes, setEdges]);

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: 12 }}>
            <div style={{ flex: 1, background: 'rgba(0,0,0,0.2)', borderRadius: 12, border: '1px solid rgba(255,255,255,0.05)', overflow: 'hidden', position: 'relative' }}>
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    nodeTypes={nodeTypes}
                    fitView
                    fitViewOptions={{ padding: 0.2 }}
                    minZoom={0.2}
                    maxZoom={2}
                    proOptions={{ hideAttribution: true }}
                >
                    <Background color="#334155" gap={24} size={1} />
                </ReactFlow>
                {nodes.length === 0 && (
                    <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569', fontSize: 11 }}>
                        Waiting for DAG initialization...
                    </div>
                )}
            </div>

            {/* Semaphore slots */}
            <div style={{ flexShrink: 0, background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 8, padding: '8px 10px' }}>
                <div style={{ fontSize: 10, color: '#64748b', marginBottom: 6 }}>
                    Parallel Semaphore: <span style={{ color: '#22d3ee', fontWeight: 700 }}>{semaphoreActive}/{semaphoreSlots}</span> slots active
                </div>
                <div style={{ display: 'flex', gap: 5 }}>
                    {Array.from({ length: semaphoreSlots }).map((_, i) => (
                        <div key={i} style={{ flex: 1, height: 8, borderRadius: 3, background: i < semaphoreActive ? 'linear-gradient(90deg,#22d3ee,#818cf8)' : 'rgba(255,255,255,0.06)', transition: 'background 0.4s', boxShadow: i < semaphoreActive ? '0 0 6px #22d3ee44' : 'none' }} />
                    ))}
                </div>
            </div>
        </div>
    );
};

export default DAGPanel;
