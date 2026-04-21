import React, { useMemo, useState } from 'react';
import { ResponsiveContainer, ComposedChart, Area, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine, Line, ReferenceArea } from 'recharts';
import { ChartEvent, DecisionType, MetricPoint, PhaseMarker } from '../types';

interface AnalyticsPanelProps {
  data: MetricPoint[];
  phaseMarkers: PhaseMarker[];
  chartEvents: ChartEvent[];
}

type MetricKey = 'tis' | 'rc' | 'stability' | 'tokenCost' | 'arDensity';

const METRIC_CONFIG: Record<MetricKey, { label: string; color: string; gradientId: string }> = {
  tis: { label: 'TIS', color: '#10b981', gradientId: 'metric-tis' },
  rc: { label: 'RC', color: '#3b82f6', gradientId: 'metric-rc' },
  stability: { label: 'Stability', color: '#f59e0b', gradientId: 'metric-stability' },
  tokenCost: { label: 'Token cost', color: '#f97316', gradientId: 'metric-token' },
  arDensity: { label: 'Accept/Reject density', color: '#a855f7', gradientId: 'metric-density' }
};

const AnalyticsPanel: React.FC<AnalyticsPanelProps> = ({ data, phaseMarkers, chartEvents }) => {
  const [metricKey, setMetricKey] = useState<MetricKey>('tis');
  const metricConfig = METRIC_CONFIG[metricKey];

  const chartData = useMemo(() => {
    const windowSize = 8;
    return data.map((point, index) => {
      const rawValue = (point as Record<string, number | undefined>)[metricKey];
      const metricValue = typeof rawValue === 'number' ? rawValue : 0;
      const window = data.slice(Math.max(0, index - windowSize + 1), index + 1);
      const windowValues = window.map(entry => {
        const value = (entry as Record<string, number | undefined>)[metricKey];
        return typeof value === 'number' ? value : 0;
      });
      const mean = windowValues.reduce((sum, value) => sum + value, 0) / (windowValues.length || 1);
      const variance = windowValues.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / (windowValues.length || 1);
      const std = Math.sqrt(variance);
      const upper = Math.min(100, metricValue + std);
      const lower = Math.max(0, metricValue - std);
      return {
        ...point,
        metricValue,
        upper,
        lower,
        acceptedMarker: point.decision === DecisionType.ACCEPT ? metricValue : null,
        rejectedMarker: point.decision === DecisionType.REJECT ? metricValue : null
      };
    });
  }, [data, metricKey]);

  const phaseBands = useMemo(() => {
    if (!phaseMarkers.length || !chartData.length) return [];
    const lastTimestamp = chartData[chartData.length - 1]?.timestamp;
    const markers = [...phaseMarkers].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
    const bands: { x1: string; x2: string; label: string; color: string }[] = [];
    for (let i = 0; i < markers.length; i += 1) {
      const current = markers[i];
      const next = markers[i + 1];
      const x1 = current.timestamp;
      const x2 = next ? next.timestamp : lastTimestamp;
      if (!x2) continue;
      const phaseColor = getPhaseColor(current.label);
      bands.push({ x1, x2, label: current.label, color: phaseColor });
    }
    return bands;
  }, [phaseMarkers, chartData]);

  const latestPoint = chartData[chartData.length - 1];
  const lastIndex = chartData.length - 1;
  const stats = useMemo(() => {
    if (!chartData.length) {
      return { latest: 0, average: 0, trend: 0 };
    }
    const latest = chartData[chartData.length - 1].metricValue;
    const window = chartData.slice(-5);
    const average = window.reduce((sum, item) => sum + item.metricValue, 0) / (window.length || 1);
    const previous = chartData.length > 1 ? chartData[chartData.length - 2].metricValue : latest;
    return { latest, average, trend: latest - previous };
  }, [chartData]);

  return (
    <div className="h-48 bg-surface border-t border-white/10 flex flex-col">
      <div className="px-4 py-2 border-b border-white/5 flex justify-between items-center">
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">Real-time Analytics</h3>
          <div className="text-[10px] text-slate-500 mt-1">
            Latest: <span className="text-slate-200 font-mono">{stats.latest.toFixed(1)}</span>
            <span className="mx-2 text-slate-600">|</span>
            Avg(5): <span className="text-slate-200 font-mono">{stats.average.toFixed(1)}</span>
            <span className={`ml-2 ${stats.trend >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {stats.trend >= 0 ? '▲' : '▼'} {Math.abs(stats.trend).toFixed(1)}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={metricKey}
            onChange={(event) => setMetricKey(event.target.value as MetricKey)}
            className="bg-slate-900/80 border border-white/10 text-[10px] text-slate-200 px-2 py-1 rounded focus:outline-none focus:ring-1 focus:ring-blue-500/50"
          >
            <option value="tis">TIS</option>
            <option value="rc">RC</option>
            <option value="stability">Stability</option>
            <option value="tokenCost">Token cost</option>
            <option value="arDensity">Accept/Reject density</option>
          </select>
          <div className="flex gap-4">
            <LegendItem color={metricConfig.color} label={metricConfig.label} />
            <LegendItem color="#22c55e" label="Accepted" />
            <LegendItem color="#ef4444" label="Rejected" />
          </div>
        </div>
      </div>
      
      <div className="flex-1 w-full p-2">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData}>
            <defs>
              <linearGradient id={metricConfig.gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={metricConfig.color} stopOpacity={0.35} />
                <stop offset="95%" stopColor={metricConfig.color} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
            <XAxis 
              dataKey="timestamp" 
              tick={{fill: '#64748b', fontSize: 10}} 
              axisLine={false} 
              tickLine={false} 
              interval={4}
            />
            <YAxis 
              hide={true} 
              domain={[0, 100]} 
            />
            <Tooltip content={({ active, payload, label }) => {
              if (!active || !payload || !payload.length) return null;
              const point = payload[0]?.payload as MetricPoint | undefined;
              const metricValue = typeof (point as any)?.metricValue === 'number'
                ? (point as any).metricValue
                : (typeof payload[0]?.value === 'number' ? payload[0].value : 0);
              return (
                <div className="bg-slate-950/95 border border-slate-700 rounded-lg px-3 py-2 text-[11px] text-slate-200 shadow-xl">
                  <div className="flex items-center justify-between text-slate-400 mb-1">
                    <span>{label}</span>
                    <span>{point?.phase || '-'}</span>
                  </div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-slate-400">{metricConfig.label}</span>
                    <span className="font-mono text-slate-100">{metricValue.toFixed(1)}</span>
                  </div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-slate-400">RC</span>
                    <span className="font-mono text-slate-100">{(point?.rc || 0).toFixed(1)}</span>
                  </div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-slate-400">Stability</span>
                    <span className="font-mono text-slate-100">{(point?.stability || 0).toFixed(1)}</span>
                  </div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-slate-400">Token cost</span>
                    <span className="font-mono text-slate-100">{(point?.tokenCost || 0).toFixed(2)}</span>
                  </div>
                  <div className="text-[10px] uppercase tracking-widest text-slate-500 mt-2">
                    {point?.decision || 'PENDING'}
                  </div>
                </div>
              );
            }} />
            {phaseBands.map((band, idx) => (
              <ReferenceArea
                key={`${band.label}-${idx}`}
                x1={band.x1}
                x2={band.x2}
                fill={band.color}
                fillOpacity={0.08}
                stroke="none"
              />
            ))}
            {phaseMarkers.map((marker, idx) => (
              <ReferenceLine
                key={`${marker.label}-${idx}`}
                x={marker.timestamp}
                stroke="#334155"
                strokeDasharray="4 4"
                label={{
                  value: marker.label,
                  position: 'insideTopRight',
                  fill: '#94a3b8',
                  fontSize: 10
                }}
              />
            ))}
            {chartEvents.map((event, idx) => (
              <ReferenceLine
                key={`${event.kind}-${idx}`}
                x={event.timestamp}
                stroke={getEventColor(event.kind)}
                strokeDasharray="2 4"
                strokeOpacity={0.6}
                label={{
                  value: event.label,
                  position: 'insideBottomRight',
                  fill: '#94a3b8',
                  fontSize: 9
                }}
              />
            ))}
            <Area
              type="monotone"
              dataKey={(entry: any) => [entry.lower, entry.upper]}
              stroke="none"
              fill={metricConfig.color}
              fillOpacity={0.08}
              isRange={true}
              isAnimationActive={false}
            />
            <Area 
              type="monotone" 
              dataKey="metricValue" 
              stroke={metricConfig.color} 
              fillOpacity={1} 
              fill={`url(#${metricConfig.gradientId})`} 
              strokeWidth={2}
              isAnimationActive={true}
              animationDuration={700}
              animationEasing="ease-out"
            />
            <Line
              type="monotone"
              dataKey="acceptedMarker"
              stroke="transparent"
              dot={(props) => renderDecisionDot(props, '#22c55e', lastIndex)}
              activeDot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="rejectedMarker"
              stroke="transparent"
              dot={(props) => renderDecisionDot(props, '#ef4444', lastIndex)}
              activeDot={false}
              isAnimationActive={false}
            />
            {latestPoint && (
              <ReferenceLine
                x={latestPoint.timestamp}
                stroke="#38bdf8"
                strokeOpacity={0.25}
                strokeWidth={2}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

const LegendItem = ({ color, label }: { color: string; label: string }) => (
  <div className="flex items-center gap-1.5">
    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
    <span className="text-[10px] text-slate-400">{label}</span>
  </div>
);

const getPhaseColor = (label: string) => {
  const key = label.toLowerCase();
  if (key.includes('research')) return '#0ea5e9';
  if (key.includes('analysis')) return '#6366f1';
  if (key.includes('draft')) return '#22c55e';
  if (key.includes('review')) return '#f59e0b';
  return '#94a3b8';
};

const getEventColor = (kind: ChartEvent['kind']) => {
  if (kind === 'role') return '#22c55e';
  if (kind === 'plan') return '#38bdf8';
  if (kind === 'phase') return '#94a3b8';
  return '#64748b';
};

const renderDecisionDot = (props: any, color: string, lastIndex: number) => {
  const { cx, cy, index } = props;
  if (typeof cx !== 'number' || typeof cy !== 'number') return null;
  const isLatest = index === lastIndex;
  return (
    <circle
      cx={cx}
      cy={cy}
      r={4}
      fill={color}
      stroke="#0f172a"
      strokeWidth={1.5}
      className={isLatest ? 'animate-pulse' : undefined}
    />
  );
};

export default AnalyticsPanel;
