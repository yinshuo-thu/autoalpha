import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/HoverCard';
import { Badge } from '@/components/ui/Badge';
import { RealtimeMetrics } from '@/types';
import { formatNumber } from '@/utils';

interface FactorListProps {
  metrics: RealtimeMetrics | null;
}

const GateBadge: React.FC<{ ok: boolean; label: string }> = ({ ok, label }) => (
  <span
    className={`rounded-full border px-2 py-0.5 text-[11px] ${
      ok
        ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
        : 'border-rose-500/30 bg-rose-500/10 text-rose-400'
    }`}
  >
    {label}
  </span>
);

export const FactorList: React.FC<FactorListProps> = ({ metrics }) => {
  const factors = metrics?.top10Factors || [];

  return (
    <Card className="glass card-hover animate-fade-in-up w-full border border-border/60">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-amber-500 animate-pulse" />
            当前因子库 Score Top 10
          </div>
          <div className="text-xs text-muted-foreground">
            先按可提交状态排序，再按 Score 排序
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/60">
                <th className="py-3 px-4 text-left font-medium text-muted-foreground">因子名</th>
                <th className="py-3 px-4 text-left font-medium text-muted-foreground">分类</th>
                <th className="py-3 px-4 text-right font-medium text-muted-foreground">Score</th>
                <th className="py-3 px-4 text-right font-medium text-muted-foreground">IC</th>
                <th className="py-3 px-4 text-right font-medium text-muted-foreground">IR</th>
                <th className="py-3 px-4 text-right font-medium text-muted-foreground">TVR</th>
                <th className="py-3 px-4 text-left font-medium text-muted-foreground">状态</th>
              </tr>
            </thead>
            <tbody>
              {factors.length > 0 ? (
                factors.map((factor, index) => (
                  <HoverCard key={`${factor.factorName}-${index}`} openDelay={180}>
                    <HoverCardTrigger asChild>
                      <tr className="border-b border-border/40 hover:bg-secondary/20 transition-colors cursor-help align-top">
                        <td className="py-3 px-4">
                          <div className="font-medium text-foreground">{factor.factorName}</div>
                          <div className="mt-1 max-w-[280px] truncate font-mono text-[11px] text-muted-foreground">
                            {factor.factorExpression}
                          </div>
                        </td>
                        <td className="py-3 px-4">
                          <Badge
                            className={
                              factor.submissionReadyFlag
                                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                : factor.passGates
                                ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                                : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                            }
                          >
                            {factor.classification || 'Unknown'}
                          </Badge>
                        </td>
                        <td className="py-3 px-4 text-right font-mono font-semibold text-amber-400">
                          {formatNumber(factor.score ?? 0, 4)}
                        </td>
                        <td className="py-3 px-4 text-right font-mono">{formatNumber(factor.ic ?? 0, 4)}</td>
                        <td className="py-3 px-4 text-right font-mono">{formatNumber(factor.icir ?? 0, 4)}</td>
                        <td className="py-3 px-4 text-right font-mono">{formatNumber(factor.turnover ?? 0, 3)}</td>
                        <td className="py-3 px-4">
                          <div className="flex flex-wrap gap-2">
                            <GateBadge ok={factor.passGates} label="Quality Gate" />
                            <GateBadge ok={factor.submissionReadyFlag} label="可提交" />
                          </div>
                        </td>
                      </tr>
                    </HoverCardTrigger>
                    <HoverCardContent
                      className="w-[460px] glass-strong p-4 shadow-xl border border-border/60"
                      side="top"
                      align="center"
                      sideOffset={8}
                    >
                      <div className="space-y-4">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="text-base font-semibold text-foreground">{factor.factorName}</div>
                            <div className="mt-1 text-xs text-muted-foreground">{factor.recommendation || factor.reason || '无额外说明'}</div>
                          </div>
                          <div className="rounded-xl bg-amber-500/10 px-3 py-2 text-right border border-amber-500/20">
                            <div className="text-[11px] text-muted-foreground">Score</div>
                            <div className="text-lg font-semibold text-amber-400">{formatNumber(factor.score ?? 0, 4)}</div>
                          </div>
                        </div>

                        <div className="rounded-xl bg-secondary/25 p-3 font-mono text-xs break-all border border-border/40">
                          {factor.factorExpression}
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                          <div className="rounded-xl bg-background/60 p-3 border border-border/40">
                            <div className="text-[11px] text-muted-foreground">IC / IR</div>
                            <div className="mt-1 text-sm font-semibold">
                              {formatNumber(factor.ic ?? 0, 4)} / {formatNumber(factor.icir ?? 0, 4)}
                            </div>
                          </div>
                          <div className="rounded-xl bg-background/60 p-3 border border-border/40">
                            <div className="text-[11px] text-muted-foreground">Turnover / RankIC</div>
                            <div className="mt-1 text-sm font-semibold">
                              {formatNumber(factor.turnover ?? 0, 3)} / {formatNumber(factor.rankIc ?? 0, 4)}
                            </div>
                          </div>
                        </div>

                        <div className="flex flex-wrap gap-2">
                          <GateBadge ok={Boolean(factor.gatesDetail?.IC)} label="IC" />
                          <GateBadge ok={Boolean(factor.gatesDetail?.IR)} label="IR" />
                          <GateBadge ok={Boolean(factor.gatesDetail?.Turnover)} label="TVR" />
                          <GateBadge ok={Boolean(factor.gatesDetail?.Concentration)} label="集中度" />
                          <GateBadge ok={Boolean(factor.gatesDetail?.Coverage)} label="Coverage" />
                          <GateBadge ok={Boolean(factor.gatesDetail?.SubmissionFormat)} label="Format" />
                        </div>

                        {factor.submissionPath && (
                          <div className="rounded-xl bg-emerald-500/5 p-3 border border-emerald-500/20">
                            <div className="text-[11px] text-muted-foreground mb-1">提交文件</div>
                            <div className="text-xs break-all text-foreground">{factor.submissionPath}</div>
                          </div>
                        )}
                      </div>
                    </HoverCardContent>
                  </HoverCard>
                ))
              ) : (
                <tr>
                  <td colSpan={7} className="py-10 text-center text-muted-foreground">
                    暂无因子数据
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
};
