import React from 'react';
import { AppPhase, RunMetrics, WorkflowPhase } from '../types';
import { Icons } from './icons';

interface RunControlProps {
  appPhase: AppPhase;
  currentWorkflowPhase: WorkflowPhase;
  metrics: RunMetrics;
}

const PHASES = [WorkflowPhase.RESEARCH, WorkflowPhase.ANALYSIS, WorkflowPhase.DRAFT, WorkflowPhase.REVIEW];

const RunControl: React.FC<RunControlProps> = ({ appPhase, currentWorkflowPhase, metrics }) => {
  const showRoutingMetrics = appPhase === AppPhase.RUNNING || appPhase === AppPhase.COMPLETED;
  return (
    <div className="flex flex-col h-full bg-surface border-l lg:border-l border-t lg:border-t-0 border-white/10 overflow-hidden w-full lg:w-80">
      
      {/* Workflow Tracker */}
      <div className="p-4 border-b border-white/10">
        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2 mb-4">
          <Icons.Activity size={16} /> Workflow Phase
        </h2>
        <div className="space-y-4">
          {PHASES.map((phase, idx) => {
            const isActive = phase === currentWorkflowPhase && appPhase === AppPhase.RUNNING;
            const isPast = PHASES.indexOf(currentWorkflowPhase) > idx || appPhase === AppPhase.COMPLETED;
            
            return (
              <div key={phase} className={`relative flex items-center gap-3 transition-all ${isActive ? 'opacity-100' : 'opacity-40'}`}>
                <div className={`
                  w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold border
                  ${isActive ? 'bg-blue-500 border-blue-400 text-white shadow-[0_0_10px_rgba(59,130,246,0.5)]' : 
                    isPast ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-400' : 
                    'bg-slate-800 border-slate-700 text-slate-500'}
                `}>
                  {isPast ? <Icons.Check size={12} /> : idx + 1}
                </div>
                <div className="flex-1">
                  <span className={`text-sm font-medium ${isActive ? 'text-white' : 'text-slate-400'}`}>{phase}</span>
                </div>
                {isActive && <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />}
                
                {/* Connector Line */}
                {idx < PHASES.length - 1 && (
                  <div className={`absolute left-3 top-6 w-px h-6 bg-slate-800 -z-10`} />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Routing Panel */}
      <div className="p-4 border-b border-white/10">
        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2 mb-4">
          <Icons.Brain size={16} /> Routing & Selection
        </h2>
        <div className="space-y-3">
          <MetricRow label="LDCL Score" value={metrics.ldcl.toFixed(3)} />
          <MetricRow label="GNE Score" value={metrics.gne.toFixed(3)} />
          <MetricRow label="MLAS Prob." value={(metrics.mlasProbability * 100).toFixed(1) + '%'} />
          <div className="flex justify-between items-center text-xs">
            <span className="text-slate-500">Diversity Check</span>
            {showRoutingMetrics ? (
              <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${metrics.diversityCheck ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                {metrics.diversityCheck ? 'PASS' : 'FAIL'}
              </span>
            ) : (
              <span className="text-[10px] text-slate-600">&nbsp;</span>
            )}
          </div>
        </div>
      </div>

      {/* Tier-0 Metrics */}
      <div className="p-4 flex-1 overflow-y-auto">
        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2 mb-4">
          <Icons.Zap size={16} /> Tier-0 Metrics
        </h2>
        <div className="space-y-4">
          <BarMetric label="Semantic Density (SD)" value={metrics.tier0.sd} color="bg-blue-500" />
          <BarMetric label="Role Coherence (RC)" value={metrics.tier0.rc} color="bg-violet-500" />
          <BarMetric label="Info Synthesis (IS)" value={metrics.tier0.is} color="bg-emerald-500" />
          <BarMetric label="Stability" value={metrics.tier0.stability} color="bg-amber-500" />
          
          <div className="mt-4 pt-4 border-t border-white/10">
            <div className="flex justify-between items-end mb-1">
              <span className="text-xs text-slate-400">Calculated TIS</span>
              <span className="text-xl font-bold text-white font-mono">
                {((metrics.tier0.sd + metrics.tier0.rc + metrics.tier0.is) / 3).toFixed(1)}
              </span>
            </div>
            <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-blue-500 to-emerald-400 transition-all duration-500" 
                style={{ width: `${(metrics.tier0.sd + metrics.tier0.rc + metrics.tier0.is) / 3}%` }} 
              />
            </div>
          </div>
        </div>
      </div>

    </div>
  );
};

const MetricRow = ({ label, value }: { label: string, value: string }) => (
  <div className="flex justify-between items-center text-xs">
    <span className="text-slate-500">{label}</span>
    <span className="font-mono text-slate-200">{value}</span>
  </div>
);

const BarMetric = ({ label, value, color }: { label: string, value: number, color: string }) => (
  <div>
    <div className="flex justify-between items-center mb-1">
      <span className="text-[10px] uppercase text-slate-500 font-semibold">{label}</span>
      <span className="text-xs font-mono text-slate-300">{value.toFixed(0)}</span>
    </div>
    <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
      <div 
        className={`h-full ${color} transition-all duration-700 ease-out`} 
        style={{ width: `${value}%` }} 
      />
    </div>
  </div>
);

export default RunControl;
