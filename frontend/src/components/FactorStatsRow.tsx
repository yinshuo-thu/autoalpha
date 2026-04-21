import React from 'react';
import { Layers, Trophy, ShieldCheck, BadgeCheck, Gauge, BarChart3 } from 'lucide-react';
import { RealtimeMetrics } from '@/types';
import { formatNumber } from '@/utils';

interface FactorStatsRowProps {
  metrics: RealtimeMetrics | null;
  onBacktest?: () => void;
}

export const FactorStatsRow: React.FC<FactorStatsRowProps> = ({ metrics, onBacktest }) => {
  const StatCard = ({
    icon: Icon,
    label,
    value,
    hint,
    accent,
  }: {
    icon: any;
    label: string;
    value: string | number;
    hint?: string;
    accent: string;
  }) => (
    <div className="glass rounded-2xl border border-border/50 p-4 card-hover min-h-[138px] flex flex-col justify-between">
      <div className="flex items-start justify-between gap-3">
        <div className={`rounded-xl p-2 ${accent}`}>
          <Icon className="h-5 w-5" />
        </div>
        {hint && <div className="text-[11px] text-muted-foreground text-right leading-5">{hint}</div>}
      </div>
      <div>
        <div className="text-xs uppercase tracking-[0.16em] text-muted-foreground">{label}</div>
        <div className="mt-2 break-words font-mono text-[clamp(1.25rem,1.8vw,1.5rem)] font-semibold leading-tight text-foreground">{value}</div>
      </div>
    </div>
  );

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4 animate-fade-in-up w-full">
      <StatCard
        icon={Layers}
        label="因子总数"
        value={metrics?.totalFactors ?? 0}
        hint="当前库内候选"
        accent="bg-primary/10 text-primary"
      />
      <StatCard
        icon={Trophy}
        label="最佳 Score"
        value={formatNumber(metrics?.score ?? 0, 4)}
        hint={metrics?.factorName || '等待结果'}
        accent="bg-amber-500/10 text-amber-500"
      />
      <StatCard
        icon={ShieldCheck}
        label="通过 Gate"
        value={metrics?.passGatesCount ?? 0}
        hint="IC / IR / TVR / 集中度"
        accent="bg-emerald-500/10 text-emerald-500"
      />
      <StatCard
        icon={BadgeCheck}
        label="可直接提交"
        value={metrics?.submissionReadyCount ?? 0}
        hint="格式和覆盖率已校验"
        accent="bg-sky-500/10 text-sky-500"
      />
      <StatCard
        icon={Gauge}
        label="最佳 Turnover"
        value={formatNumber(metrics?.turnover ?? 0, 4)}
        hint="越低越有利于 Score"
        accent="bg-rose-500/10 text-rose-500"
      />
      <button
        type="button"
        className="glass rounded-2xl border border-primary/20 p-4 min-h-[138px] flex flex-col justify-center items-center gap-3 text-center hover:scale-[1.02] transition-all"
        onClick={onBacktest}
      >
        <div className="rounded-full bg-primary p-4 text-primary-foreground shadow-lg">
          <BarChart3 className="h-6 w-6" />
        </div>
        <div>
          <div className="text-sm font-semibold text-foreground">一键回测</div>
          <div className="text-xs text-muted-foreground mt-1">对当前最优因子做独立策略复核</div>
        </div>
      </button>
    </div>
  );
};
