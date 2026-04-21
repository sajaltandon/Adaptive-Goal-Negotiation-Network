import React, { useState, useEffect, useMemo, useRef } from 'react';
import { ActiveAgent, AgentModel, AppPhase, Message, DecisionType, WorkflowPhase, TeamFormationData, TeamPlanData } from '../types';
import { Icons } from './icons';

interface WorkspaceProps {
  appPhase: AppPhase;
  selectedAgents: ActiveAgent[];
  availableModels: AgentModel[];
  onAddAgent: (agent: AgentModel) => void;
  onRemoveAgent: (id: string) => void;
  onConfirmTeam: () => void;
  onStartRun: (prompt: string) => void;
  messages: Message[];
  workflowPhase: WorkflowPhase;
  teamFormation: TeamFormationData | null;
  teamPlan: TeamPlanData | null;
  isAutoPopulating?: boolean;
  autoPopulateModels?: AgentModel[];
}

const Workspace: React.FC<WorkspaceProps> = ({
  appPhase,
  selectedAgents,
  availableModels,
  onAddAgent,
  onRemoveAgent,
  onConfirmTeam,
  onStartRun,
  messages,
  workflowPhase,
  teamFormation,
  teamPlan,
  isAutoPopulating = false,
  autoPopulateModels = []
}) => {
  const [prompt, setPrompt] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat
  useEffect(() => {
    if (chatBottomRef.current) {
      chatBottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Drag Handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const modelId = e.dataTransfer.getData('modelId');
    const model = availableModels.find(m => m.id === modelId);
    if (model) {
      // Check if already exists to prevent duplicates if desired, or allow multiples
      // For this demo, let's allow unique IDs only
      const isAlreadyAdded = selectedAgents.some(a => a.id.startsWith(model.id));
      if (!isAlreadyAdded) {
         onAddAgent(model);
      }
    }
  };

  const handleDragStart = (e: React.DragEvent, modelId: string) => {
    e.dataTransfer.setData('modelId', modelId);
  };

  // Views
  if (appPhase === AppPhase.SETUP) {
    return (
      <div className="flex flex-col h-full relative overflow-y-auto">
        {/* Model Bay */}
        <div className="p-3 md:p-4 bg-surfaceHighlight/30 border-b border-white/5 flex-shrink-0">
           <h2 className="text-base font-semibold text-white mb-3 flex items-center gap-2">
             <Icons.Database className="text-blue-400" size={20} /> Model Bay
           </h2>
           <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-thin">
             {availableModels.map(model => (
               <div 
                  key={model.id}
                  draggable
                  onDragStart={(e) => handleDragStart(e, model.id)}
                  className="flex-shrink-0 w-40 bg-surface border border-white/10 p-2 rounded-lg cursor-grab active:cursor-grabbing hover:border-blue-500/50 hover:bg-blue-900/10 transition-all group"
               >
                  <div className="flex items-center gap-2 mb-2">
                    <div className={`w-5 h-5 rounded ${model.avatarColor} opacity-80`} />
                    <span className="font-bold text-[13px] text-slate-200">{model.name}</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    <span className="text-[10px] bg-slate-800 text-slate-500 px-1.5 py-0.5 rounded border border-white/5">{model.provider}</span>
                    {model.capabilities.map(cap => (
                      <span key={cap} className="text-[10px] bg-slate-800 px-1.5 py-0.5 rounded text-slate-400">{cap}</span>
                    ))}
                  </div>
               </div>
             ))}
           </div>
        </div>

        {/* Drop Zone */}
        <div className="flex-1 p-4 md:p-8 flex items-center justify-center">
          <div 
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`
              w-full max-w-3xl min-h-[18rem] h-[45vh] lg:h-96 rounded-2xl border-2 border-dashed flex flex-col items-center justify-center transition-all duration-300 relative
              ${isDragOver ? 'border-blue-500 bg-blue-900/20 scale-105' : 'border-slate-700 bg-surface/50'}
            `}
          >
            {isAutoPopulating && autoPopulateModels.length > 0 && (
              <div className="absolute inset-0 pointer-events-none overflow-hidden">
                {autoPopulateModels.map((model, idx) => (
                  <div
                    key={`${model.id}-ghost`}
                    className="absolute left-1/2 -translate-x-1/2 top-6 w-44 bg-surface border border-white/20 p-2 rounded-lg shadow-xl animate-auto-drag"
                    style={{ animationDelay: `${idx * 180}ms` }}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <div className={`w-4 h-4 rounded ${model.avatarColor} opacity-80`} />
                      <span className="font-semibold text-[12px] text-slate-200 truncate">{model.name}</span>
                    </div>
                    <div className="text-[10px] text-slate-500">Auto-docking...</div>
                  </div>
                ))}
              </div>
            )}

            {selectedAgents.length === 0 ? (
              <div className="text-center p-10 pointer-events-none">
                <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4 text-slate-500">
                  <Icons.Users size={32} />
                </div>
                <h3 className="text-xl font-medium text-slate-300 mb-2">Autonomous Team Setup</h3>
                <p className="text-slate-500">Connect LM Studio. AGNN will auto-discover and auto-place the team.</p>
              </div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 w-full p-6 md:p-8 overflow-y-auto max-h-full">
                {selectedAgents.map(agent => (
                  <div key={agent.id} className={`relative bg-surfaceHighlight p-4 rounded-xl border border-white/10 flex flex-col items-center gap-2 group ${agent.autoAdded ? "animate-auto-dock" : "animate-slide-up"}` }>
                    <button 
                      onClick={() => onRemoveAgent(agent.id)}
                      className="absolute top-2 right-2 text-slate-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <Icons.X size={16} />
                    </button>
                    <div className={`w-10 h-10 rounded-lg ${agent.avatarColor} flex items-center justify-center text-white font-bold`}>
                      {agent.name.substring(0,2)}
                    </div>
                    <span className="font-medium text-slate-200">{agent.name}</span>
                    <span className="text-xs text-slate-500">{agent.provider}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Action Bar */}
        <div className="p-4 md:p-6 border-t border-white/10 flex justify-end bg-surface/95 backdrop-blur flex-shrink-0 sticky bottom-0">
           <button 
             disabled={selectedAgents.length < 2}
             onClick={onConfirmTeam}
             className={`
               flex items-center gap-2 px-6 py-3 rounded-lg font-bold transition-all
               ${selectedAgents.length < 2 
                 ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                 : 'bg-blue-600 text-white hover:bg-blue-500 hover:shadow-lg hover:shadow-blue-500/20'}
             `}
           >
             Continue to Prompt <Icons.ChevronRight size={18} />
           </button>
        </div>
      </div>
    );
  }

  if (appPhase === AppPhase.NEGOTIATION) {
    return (
      <RoleNegotiationView onComplete={(p) => onStartRun(p)} initialPrompt={prompt} setPrompt={setPrompt} />
    );
  }

  // RUNNING & COMPLETED Views
  return (
    <div className="flex flex-col h-full bg-background relative">
      
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6 scrollbar-thin">
        <TeamFormationPanel teamFormation={teamFormation} teamPlan={teamPlan} appPhase={appPhase} />
        {messages.map((msg, index) => {
          if (msg.isPhaseChange) {
            return (
              <div key={msg.id} className="flex items-center justify-center my-6">
                 <div className="bg-slate-800/80 border border-slate-700 rounded-full px-4 py-1.5 text-xs text-slate-300 font-mono tracking-wider flex items-center gap-2 backdrop-blur-sm">
                   <Icons.Activity size={12} className="text-blue-400" />
                   {msg.content}
                 </div>
              </div>
            );
          }

          const isSystem = msg.agentId === 'SYSTEM';

          return (
            <div key={msg.id} className={`flex gap-4 max-w-4xl mx-auto animate-fade-in ${isSystem ? 'opacity-70' : ''}`}>
              {!isSystem && (
                <div className="flex-shrink-0 mt-1">
                   <div className={`w-8 h-8 rounded-md flex items-center justify-center text-white text-xs font-bold shadow-lg ${msg.agentId === 'SYSTEM' ? 'bg-slate-700' : selectedAgents.find(a => a.id === msg.agentId)?.avatarColor || 'bg-slate-700'}`}>
                     {isSystem ? 'SYS' : msg.agentName?.substring(0,2)}
                   </div>
                </div>
              )}
              
              <div className={`flex-1 ${isSystem ? 'text-center' : ''}`}>
                {!isSystem && (
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-bold text-slate-200 text-sm">{msg.agentName}</span>
                    <span className="text-[10px] bg-slate-800 text-slate-400 px-1.5 rounded border border-white/5">{msg.role}</span>
                    <span className="text-[10px] text-slate-600 ml-auto font-mono">
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                )}
                
                <div className={`
                  p-4 rounded-lg text-sm leading-relaxed border
                  ${isSystem ? 'bg-transparent border-transparent text-slate-500 italic text-center' : 'bg-surfaceHighlight/40 border-white/5 text-slate-300 shadow-sm'}
                `}>
                  {msg.content}
                </div>

                {!isSystem && (
                  <div className="mt-2 flex gap-2">
                    <Badge type={msg.decision} />
                    <span className="text-[10px] text-slate-600 border border-slate-800 px-1.5 rounded bg-slate-900/50 uppercase tracking-widest">{msg.phase}</span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
        <div ref={chatBottomRef} />
      </div>

      {/* Task Bar (Input - Disabled during run for this demo, or allows injection) */}
      <div className="p-4 bg-surface border-t border-white/10 flex justify-center">
        <div className="w-full max-w-3xl relative">
          <input 
            type="text" 
            placeholder={appPhase === AppPhase.COMPLETED ? "Run complete. Export logs or start new." : "System running... (Input disabled in demo mode)"}
            disabled={true}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-4 pr-12 py-3 text-sm text-slate-400 focus:outline-none focus:border-blue-500/50 transition-colors"
          />
          <div className="absolute right-2 top-2 p-1.5 bg-slate-800 rounded text-slate-500">
             <Icons.Zap size={16} />
          </div>
        </div>
      </div>
    </div>
  );
};

const TeamFormationPanel = ({ teamFormation, teamPlan, appPhase }: { teamFormation: TeamFormationData | null; teamPlan: TeamPlanData | null; appPhase: AppPhase }) => {
  const roles = useMemo(() => {
    if (!teamFormation) return [];
    if (teamFormation.roles && teamFormation.roles.length) {
      return teamFormation.roles;
    }
    const roleMap = new Map<string, TeamFormationData['members'][number]['role']>();
    teamFormation.members.forEach(member => {
      if (member.role?.name) {
        roleMap.set(member.role.name, member.role);
      }
      (member.secondary_roles || []).forEach(role => {
        if (role?.name) {
          roleMap.set(role.name, role);
        }
      });
    });
    return Array.from(roleMap.values());
  }, [teamFormation]);

  const negotiationByTurn = useMemo(() => {
    if (!teamFormation?.negotiation?.length) return [];
    const grouped = new Map<number, TeamFormationData['negotiation']>();
    teamFormation.negotiation.forEach(entry => {
      const list = grouped.get(entry.turn) || [];
      list.push(entry);
      grouped.set(entry.turn, list);
    });
    return Array.from(grouped.entries()).sort((a, b) => a[0] - b[0]);
  }, [teamFormation]);

  const [showRolesPanel, setShowRolesPanel] = useState(false);
  const [showNegotiationPanel, setShowNegotiationPanel] = useState(false);
  const [showConsensusPanel, setShowConsensusPanel] = useState(false);

  useEffect(() => {
    if (appPhase === AppPhase.RUNNING || appPhase === AppPhase.COMPLETED) {
      setShowRolesPanel(true);
    } else {
      setShowRolesPanel(false);
      setShowNegotiationPanel(false);
      setShowConsensusPanel(false);
    }
  }, [appPhase]);

  useEffect(() => {
    if (roles.length > 0 && !showNegotiationPanel) {
      const timer = setTimeout(() => setShowNegotiationPanel(true), 250);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [roles.length, showNegotiationPanel]);

  useEffect(() => {
    const shouldReveal = (teamFormation?.negotiation?.length || teamFormation?.members?.length || teamPlan);
    if (shouldReveal && !showConsensusPanel) {
      const timer = setTimeout(() => setShowConsensusPanel(true), 350);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [teamFormation?.negotiation?.length, teamFormation?.members?.length, teamPlan, showConsensusPanel]);

  if (!showRolesPanel) {
    return null;
  }

  return (
    <div className="bg-surface/70 border border-white/10 rounded-2xl p-4 mb-6 shadow-lg shadow-black/20">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div>
          <div className="text-xs uppercase tracking-[0.25em] text-slate-500">Tier-1</div>
          <h3 className="text-lg font-semibold text-white">Team Formation & Role Negotiation</h3>
        </div>
        {teamFormation && (
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Icons.Clock size={14} className="text-emerald-400" />
            Negotiation turns: {teamFormation.formationTurns}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <section className="bg-slate-950/40 border border-white/5 rounded-xl p-3 animate-fade-in flex flex-col xl:h-[26rem]">
          <div className="text-xs uppercase tracking-widest text-slate-500 mb-3">Generated Roles</div>
          <div className="space-y-3 flex-1 min-h-0 overflow-y-auto pr-2 scrollbar-thin">
            {roles.length === 0 && (
              <div className="space-y-3">
                {[0, 1, 2].map(idx => (
                  <div key={idx} className="border border-white/5 rounded-lg p-3 bg-slate-900/40 animate-pulse">
                    <div className="h-4 w-32 bg-slate-800 rounded" />
                    <div className="h-3 w-48 bg-slate-800 rounded mt-2" />
                    <div className="flex gap-2 mt-3">
                      <div className="h-6 w-20 bg-slate-800 rounded-full" />
                      <div className="h-6 w-24 bg-slate-800 rounded-full" />
                    </div>
                  </div>
                ))}
              </div>
            )}
            {roles.map((role, index) => (
              <div
                key={role.name}
                className="border border-white/5 rounded-lg p-3 bg-slate-900/50 animate-slide-up"
                style={{ animationDelay: `${index * 120}ms` }}
              >
                <div className="flex items-center justify-between">
                  <div className="font-semibold text-slate-200">{role.name}</div>
                  <span className="text-[10px] uppercase tracking-widest text-blue-400">Role</span>
                </div>
                <div className="text-xs text-slate-500 mt-1">{role.description}</div>
                <div className="flex flex-wrap gap-1 mt-2">
                  {role.responsibilities?.map(item => (
                    <span key={item} className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-white/5">
                      {item}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>

        {showNegotiationPanel && (
          <section className="bg-slate-950/40 border border-white/5 rounded-xl p-3 animate-fade-in flex flex-col xl:h-[26rem]">
            <div className="text-xs uppercase tracking-widest text-slate-500 mb-3">Negotiation Transcript</div>
            <div className="flex-1 min-h-0 overflow-y-auto pr-2 space-y-3 scrollbar-thin">
              {negotiationByTurn.length === 0 && (
                <div className="text-sm text-slate-500">Negotiation is starting... waiting for first proposals.</div>
              )}
              {negotiationByTurn.map(([turn, entries]) => (
                <div key={turn} className="space-y-3">
                  <div className="text-[10px] uppercase tracking-widest text-slate-600">Turn {turn}</div>
                  {entries.map((entry, idx) => (
                    <div
                      key={`${entry.agent}-${idx}`}
                      className="py-2 animate-slide-up"
                      style={{ animationDelay: `${idx * 80}ms` }}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-[10px] uppercase tracking-widest text-blue-400">{entry.agent}</span>
                        {(entry.proposed_role || entry.final_role) && (
                          <span className="text-[10px] px-2 py-0.5 rounded-full border border-white/10 bg-slate-800 text-slate-300">
                            {entry.final_role || entry.proposed_role}
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-slate-300 leading-relaxed">{entry.content}</div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </section>
        )}

        {showConsensusPanel && (
          <section className="bg-slate-950/40 border border-white/5 rounded-xl p-3 animate-fade-in flex flex-col xl:h-[26rem]">
            <div className="text-xs uppercase tracking-widest text-slate-500 mb-3">Consensus & Plan</div>
            <div className="space-y-3 flex-1 min-h-0 overflow-y-auto pr-2 scrollbar-thin">
              {(teamFormation?.members || []).map((member, index) => (
                <div
                  key={member.agent_id}
                  className="border border-white/5 rounded-lg p-3 bg-slate-900/50 animate-slide-up"
                  style={{ animationDelay: `${index * 120}ms` }}
                >
                  <div className="flex items-center justify-between">
                    <div className="font-semibold text-slate-200">{member.agent_id}</div>
                    <span className="text-[10px] text-emerald-400">Conf. {member.confidence.toFixed(2)}</span>
                  </div>
                  <div className="text-xs text-slate-400 mt-1">{member.role?.name}</div>
                  {teamPlan?.responsibilities?.[member.agent_id]?.length ? (
                    <div className="mt-2">
                      <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">Responsibilities</div>
                      <div className="flex flex-wrap gap-1">
                        {teamPlan.responsibilities[member.agent_id].map(item => (
                          <span key={item} className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null}
                  {teamPlan?.avoid?.[member.agent_id]?.length ? (
                    <div className="mt-2">
                      <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">Avoid</div>
                      <div className="flex flex-wrap gap-1">
                        {teamPlan.avoid[member.agent_id].map(item => (
                          <span key={item} className="text-[10px] px-2 py-0.5 rounded-full bg-red-500/10 text-red-300 border border-red-500/20">
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              ))}
              {!teamFormation?.members?.length && (
                <div className="text-sm text-slate-500">Consensus will appear after negotiation completes.</div>
              )}
            </div>
          </section>
        )}
      </div>
    </div>
  );
};

// Sub-component for Role Negotiation
const RoleNegotiationView = ({ onComplete, initialPrompt, setPrompt }: { 
  onComplete: (prompt: string) => void, 
  initialPrompt: string,
  setPrompt: (s: string) => void
}) => {
  const [stage, setStage] = useState<'PROMPT' | 'LAUNCHING'>('PROMPT');

  if (stage === 'PROMPT') {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 animate-fade-in">
        <div className="w-full max-w-2xl">
          <h2 className="text-2xl font-bold text-white mb-2">Define the Objective</h2>
          <p className="text-slate-400 mb-6">Describe the complex task for the AGNN swarm to solve.</p>
          
          <textarea 
            value={initialPrompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g., Conduct a comprehensive analysis of the potential impact of quantum computing on modern cryptography protocols by 2030..."
            className="w-full h-40 bg-surfaceHighlight/50 border border-white/10 rounded-xl p-4 text-slate-200 focus:outline-none focus:border-blue-500/50 resize-none mb-6"
          />

          <div className="flex justify-end">
             <button 
               onClick={() => {
                 setStage('LAUNCHING');
                 onComplete(initialPrompt);
               }}
               disabled={!initialPrompt.trim()}
               className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2.5 rounded-lg font-medium transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
             >
               Start Negotiation <Icons.BrainCircuit size={18} />
             </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center h-full p-8 relative overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-blue-900/10 via-background to-background" />
      <div className="relative z-10 text-center max-w-xl">
        <h2 className="text-2xl font-bold text-white mb-2">Launching Tier-1 Formation</h2>
        <p className="text-slate-400 mb-6">AGNN is generating roles and starting the negotiation loop.</p>
        <div className="flex items-center justify-center gap-2 text-blue-400 animate-pulse">
          <Icons.Play size={18} /> Connecting to agents...
        </div>
      </div>
    </div>
  );
};

const Badge = ({ type }: { type?: DecisionType }) => {
  if (!type) return null;
  
  const styles = {
    [DecisionType.ACCEPT]: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    [DecisionType.REWRITE]: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    [DecisionType.REJECT]: 'bg-red-500/10 text-red-400 border-red-500/20',
    [DecisionType.INFO]: 'bg-slate-500/10 text-slate-400 border-slate-500/20',
  };

  const icons = {
    [DecisionType.ACCEPT]: <Icons.Check size={10} />,
    [DecisionType.REWRITE]: <Icons.Refresh size={10} />,
    [DecisionType.REJECT]: <Icons.Alert size={10} />,
    [DecisionType.INFO]: <Icons.Activity size={10} />,
  };

  return (
    <span className={`flex items-center gap-1 text-[10px] px-2 py-0.5 rounded border font-bold uppercase ${styles[type]}`}>
      {icons[type]} {type}
    </span>
  );
};

export default Workspace;
