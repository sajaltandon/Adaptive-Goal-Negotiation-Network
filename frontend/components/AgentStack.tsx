import React, { useState } from 'react';
import { ActiveAgent, AppPhase, AgentRole } from '../types';
import { Icons } from './icons';

interface AgentStackProps {
  agents: ActiveAgent[];
  appPhase: AppPhase;
  onConnectLMStudio: (url: string) => void;
}

const AgentStack: React.FC<AgentStackProps> = ({ agents, appPhase, onConnectLMStudio }) => {
  const [lmStudioUrl, setLmStudioUrl] = useState('http://10.119.170.167:1234');
  const [isConnecting, setIsConnecting] = useState(false);

  const getRoleIcon = (role: AgentRole | string) => {
    switch (role) {
      case AgentRole.RESEARCHER: return <Icons.Search size={14} />;
      case AgentRole.CRITIC: return <Icons.Alert size={14} />;
      case AgentRole.DRAFTER: return <Icons.File size={14} />;
      default: return <Icons.Brain size={14} />;
    }
  };

  const handleConnect = () => {
    setIsConnecting(true);
    // Simulate network delay for "real" feel
    setTimeout(() => {
      onConnectLMStudio(lmStudioUrl);
      setIsConnecting(false);
    }, 800);
  };

  return (
    <div className="flex flex-col h-full bg-surface border-r lg:border-r border-b lg:border-b-0 border-white/10 overflow-hidden">
      <div className="p-4 border-b border-white/10 flex justify-between items-center flex-shrink-0">
        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
          <Icons.Users size={16} /> Agent Stack
        </h2>
        <span className="text-xs bg-slate-800 px-2 py-0.5 rounded-full text-slate-400">{agents.length} Active</span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {agents.length === 0 && (
          <div className="text-center text-slate-600 mt-10 text-sm italic">
            No agents selected. Drag models here to build your team.
          </div>
        )}

        {agents.map((agent) => (
          <div
            key={agent.id}
            className={`
              relative p-3 rounded-lg border transition-all duration-300
              ${agent.status === 'speaking'
                ? 'bg-slate-800/50 border-slate-600/60 shadow-[0_0_10px_rgba(148,163,184,0.25)]'
                : 'bg-surfaceHighlight/50 border-white/5 hover:border-white/10'}
            `}
          >
            {/* Header */}
            <div className="flex items-center gap-3 mb-2">
              <div className={`w-8 h-8 rounded-md flex items-center justify-center text-white font-bold text-xs ${agent.avatarColor} shadow-lg`}>
                {agent.name.substring(0, 2)}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex justify-between items-center">
                  <h3 className="font-semibold text-sm truncate text-slate-200">{agent.name}</h3>
                  {agent.confidence > 0 && (
                    <span className="text-[10px] text-success font-mono">
                      {(agent.confidence * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-1 text-xs text-slate-500 mt-0.5">
                  {getRoleIcon(agent.role)}
                  <span>{agent.role}</span>
                </div>
              </div>
            </div>

            {/* Stats Grid */}
            {appPhase !== AppPhase.SETUP && (
              <div className="grid grid-cols-4 gap-1 mt-3 pt-3 border-t border-white/5">
                <StatItem label="Turn" value={agent.stats.turns} />
                <StatItem label="Acc" value={agent.stats.accepted} color="text-success" />
                <StatItem label="Rej" value={agent.stats.rejects} color="text-danger" />
                <StatItem label="Rev" value={agent.stats.rewrites} color="text-warning" />
              </div>
            )}

            {/* Speaking Indicator */}
            {agent.status === 'speaking' && (
              <div className="absolute top-2 right-2 w-2 h-2 rounded-full bg-slate-400 animate-pulse-glow" />
            )}
          </div>
        ))}
      </div>

      {/* LM Studio Integration */}
      <div className="p-4 border-t border-white/10 bg-slate-950 flex-shrink-0">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-3 flex items-center gap-2">
          <Icons.Code size={12} /> Local LLM Source
        </h3>
        <div className="space-y-2">
          <input
            type="text"
            value={lmStudioUrl}
            onChange={(e) => setLmStudioUrl(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1.5 text-xs text-slate-300 placeholder-slate-600 focus:outline-none focus:border-slate-500"
            placeholder="http://10.119.170.167:1234"
          />
          <button
            onClick={handleConnect}
            disabled={isConnecting}
            className={`
                    w-full flex items-center justify-center gap-2 py-1.5 rounded text-xs font-medium border transition-all
                    ${isConnecting
                ? 'bg-slate-800 text-slate-500 border-slate-800'
                : 'bg-surfaceHighlight hover:bg-slate-700 text-primary border-primary/30 hover:border-primary/60'}
                `}
          >
            {isConnecting ? (
              <>Connecting...</>
            ) : (
              <><Icons.Refresh size={12} /> Connect LM Studio</>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

const StatItem = ({ label, value, color = "text-slate-400" }: { label: string, value: number, color?: string }) => (
  <div className="flex flex-col items-center">
    <span className="text-[9px] uppercase text-slate-600 font-bold">{label}</span>
    <span className={`text-xs font-mono font-medium ${color}`}>{value}</span>
  </div>
);

export default AgentStack;
