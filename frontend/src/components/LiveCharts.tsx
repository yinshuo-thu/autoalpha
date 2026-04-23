import React, { useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { TimeSeriesData, RealtimeMetrics, LogEntry } from '@/types';
import { formatNumber, formatPercent } from '@/utils';
import { Trophy, Activity, Gauge, ShieldCheck, Bot, FileText } from 'lucide-react';

interface LiveChartsProps {
  equityCurve: TimeSeriesData[];
  drawdownCurve: TimeSeriesData[];
  metrics: RealtimeMetrics | null;
  isRunning: boolean;
  logs: LogEntry[];
  llmMiningRecent?: Record<string, unknown>[];
  logPaths?: { researchLog?: string; llmMiningJsonl?: string };
}

function LlmRecordBlock({ rec }: { rec: Record<string, unknown> }) {
  const ev = String(rec.event ?? '?');
  const src = String(rec.mining_source ?? '');
  const when = String(rec.logged_at ?? '');
  const badge =
    ev === 'llm_ok'
      ? 'bg-emerald-500/20 text-emerald-300'
      : ev.includes('fallback')
        ? 'bg-rose-500/20 text-rose-300'
        : 'bg-sky-500/20 text-sky-300';

  if (ev === 'llm_ok') {
    const p = rec.parsed as Record<string, unknown> | undefined;
    const thought = p?.thought_process != null ? String(p.thought_process) : '';
    const formula = p?.formula != null ? String(p.formula) : '';
    return (
      <div className="border-b border-slate-800/80 pb-3 mb-3 last:mb-0 last:border-0 last:pb-0">
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
          <span className={`rounded px-1.5 py-0.5 font-medium ${badge}`}>{ev}</span>
          <span>{src || 'query'}</span>
          {rec.model != null && <span className="text-slate-600">model={String(rec.model)}</span>}
          {rec.transport != null && <span className="text-slate-600">{String(rec.transport)}</span>}
          <span className="text-slate-600">{when}</span>
        </div>
        {rec.user_prompt_excerpt != null && String(rec.user_prompt_excerpt).trim() !== '' && (
          <p className="mt-2 text-xs text-slate-500 line-clamp-4">
            提示摘要: {String(rec.user_prompt_excerpt)}
          </p>
        )}
        {formula && (
          <code className="mt-2 block rounded-lg bg-slate-900/80 p-2 text-[13px] text-cyan-300 break-all">
            {formula}
          </code>
        )}
        {thought && (
          <p className="mt-2 text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">{thought}</p>
        )}
      </div>
    );
  }

  if (ev === 'research_loop_batch') {
    const ideas = Array.isArray(rec.ideas) ? (rec.ideas as Record<string, unknown>[]) : [];
    const gen = rec.generation_mode != null ? String(rec.generation_mode) : '';
    const it = rec.iteration != null ? String(rec.iteration) : '';
    return (
      <div className="border-b border-slate-800/80 pb-3 mb-3 last:mb-0 last:border-0 last:pb-0">
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
          <span className={`rounded px-1.5 py-0.5 font-medium ${badge}`}>{ev}</span>
          <span>iter {it}</span>
          <span className="text-amber-200/90">mode={gen}</span>
          <span className="text-slate-600">{when}</span>
        </div>
        <ul className="mt-2 space-y-2">
          {ideas.map((idea, j) => (
            <li key={j} className="rounded-lg bg-slate-900/50 p-2 text-sm">
              <div className="text-[11px] text-slate-500">
                #{j + 1} source={String(idea.source ?? '?')}
              </div>
              {idea.formula != null && (
                <code className="mt-1 block text-[13px] text-cyan-300/95 break-all">{String(idea.formula)}</code>
              )}
              {(idea.rationale != null || idea.thought_process != null) && (
                <p className="mt-1 text-xs text-slate-300 leading-relaxed">
                  {String(idea.rationale ?? idea.thought_process ?? '')}
                </p>
              )}
            </li>
          ))}
        </ul>
      </div>
    );
  }

  if (ev.includes('fallback')) {
    const err = rec.error != null ? String(rec.error) : '';
    const p = rec.parsed as Record<string, unknown> | undefined;
    return (
      <div className="border-b border-slate-800/80 pb-3 mb-3 last:mb-0 last:border-0 last:pb-0">
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
          <span className={`rounded px-1.5 py-0.5 font-medium ${badge}`}>{ev}</span>
          <span>{src}</span>
          <span className="text-slate-600">{when}</span>
        </div>
        {err && <p className="mt-2 text-xs text-rose-300/95 break-all">{err}</p>}
        {p?.formula != null && (
          <code className="mt-2 block text-[12px] text-slate-400 break-all">离线公式: {String(p.formula)}</code>
        )}
      </div>
    );
  }

  return (
    <div className="border-b border-slate-800/80 pb-2 mb-2 last:mb-0 last:border-0 text-xs text-slate-500 font-mono break-all">
      <span className="text-slate-400">{ev}</span> {src} · {when}
      <pre className="mt-1 max-h-24 overflow-hidden text-[11px] text-slate-600">
        {JSON.stringify(rec, null, 0).slice(0, 400)}
        {(JSON.stringify(rec).length > 400 ? '…' : '')}
      </pre>
    </div>
  );
}

export const LiveCharts: React.FC<LiveChartsProps> = ({
  equityCurve: _equityCurve,
  drawdownCurve: _drawdownCurve,
  metrics,
  isRunning: _isRunning,
  logs,
  llmMiningRecent = [],
  logPaths,
}) => {
  const logContainerRef = useRef<HTMLDivElement>(null);
  const logEndRef = useRef<HTMLDivElement>(null);
  const isAutoScrollRef = useRef(true);

  // Handle manual scroll to toggle auto-scroll
  const handleScroll = () => {
    if (logContainerRef.current) {
      const { scrollHeight, clientHeight, scrollTop } = logContainerRef.current;
      // If user is within 50px of bottom, enable auto-scroll. Otherwise disable it.
      // Use a larger threshold (100px) to be more forgiving
      // Also ensure we handle floating point differences by checking absolute difference
      const distanceToBottom = Math.abs(scrollHeight - clientHeight - scrollTop);
      const isNearBottom = distanceToBottom < 100;
      
      // Only update if the user initiated the scroll (or if we are correcting drift)
      // This is a simple heuristic: if we are near bottom, re-enable auto-scroll
      if (isNearBottom) {
        isAutoScrollRef.current = true;
      } else {
        isAutoScrollRef.current = false;
      }
    }
  };

  useEffect(() => {
    if (isAutoScrollRef.current) {
      // Use requestAnimationFrame to ensure we scroll AFTER layout updates
      requestAnimationFrame(() => {
        if (logContainerRef.current) {
          const { scrollHeight, clientHeight } = logContainerRef.current;
          // Use scrollTo instead of scrollIntoView to avoid affecting parent containers
          logContainerRef.current.scrollTo({
            top: scrollHeight - clientHeight,
            behavior: 'smooth'
          });
        }
      });
    }
  }, [logs]);

  // When mouse leaves the container, if we are near bottom, force auto-scroll to be true
  // This helps when the user was just looking at logs and moves mouse away
  const handleMouseLeave = () => {
    if (logContainerRef.current) {
      const { scrollHeight, clientHeight, scrollTop } = logContainerRef.current;
      const distanceToBottom = Math.abs(scrollHeight - clientHeight - scrollTop);
      if (distanceToBottom < 100) {
        isAutoScrollRef.current = true;
      }
    }
  };

  const getLogIcon = (level: LogEntry['level']) => {
    switch (level) {
      case 'success': return '[OK]';
      case 'error': return '[ERR]';
      case 'warning': return '[WARN]';
      default: return '[SYS]';
    }
  };

  const getLogColor = (level: LogEntry['level']) => {
    switch (level) {
      case 'success': return 'text-green-400 font-bold';
      case 'error': return 'text-red-400 font-bold md:bg-red-950/30';
      case 'warning': return 'text-yellow-400 font-semibold';
      default: return 'text-green-500';
    }
  };

  const StatCard = ({ icon: Icon, label, value, trend, color }: any) => (
    <div className="glass rounded-xl p-4 card-hover h-[140px] flex flex-col justify-between">
      <div className="flex items-start justify-between mb-2">
        <div className={`p-2 rounded-lg ${color} bg-opacity-20`}>
          <Icon className={`h-5 w-5 ${color}`} />
        </div>
        {trend !== undefined && (
          <Badge variant={trend > 0 ? 'success' : 'destructive'} className="text-xs">
            {trend > 0 ? '+' : ''}{formatPercent(trend, 1)}
          </Badge>
        )}
      </div>
      <div className="text-xs text-muted-foreground mb-1">{label}</div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  );

  return (
    <div className="space-y-4">
      {/* Key Metrics Row */}
      {metrics && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 animate-fade-in-up">
          <StatCard
            icon={Trophy}
            label={metrics.factorName ? `最佳因子 Score (${metrics.factorName.split('_').slice(0,2).join('_')}...)` : "最佳因子 Score"}
            value={formatNumber(metrics.score ?? 0, 4)}
            color="text-amber-400"
          />
          <StatCard
            icon={Activity}
            label="最佳因子 IC"
            value={formatNumber(metrics.ic, 4)}
            color="text-primary"
          />
          <StatCard
            icon={Gauge}
            label="最佳因子 Turnover"
            value={formatNumber(metrics.turnover ?? 0, 4)}
            color="text-rose-400"
          />
          <StatCard
            icon={ShieldCheck}
            label="可直接提交数量"
            value={metrics.submissionReadyCount ?? 0}
            color="text-emerald-400"
          />
        </div>
      )}

      {/* Main Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {(llmMiningRecent.length > 0 || logPaths?.researchLog || logPaths?.llmMiningJsonl) && (
          <Card className="glass card-hover animate-fade-in-left lg:col-span-4 flex flex-col max-h-[min(480px,70vh)]">
            <CardHeader className="pb-2 shrink-0">
              <CardTitle className="text-base flex items-center gap-2">
                <Bot className="h-5 w-5 text-violet-400" />
                LLM 回复与因子指导
              </CardTitle>
              <p className="text-xs text-muted-foreground mt-1">
                数据来自服务端审计日志（与引擎写入的 mining_log.jsonl 一致）。未跑通时也可根据「错误 / 离线回退」条目排查。
              </p>
              {(logPaths?.researchLog || logPaths?.llmMiningJsonl) && (
                <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-500 font-mono">
                  {logPaths?.researchLog && (
                    <span className="flex items-center gap-1">
                      <FileText className="h-3 w-3" />
                      research: {logPaths.researchLog}
                    </span>
                  )}
                  {logPaths?.llmMiningJsonl && (
                    <span className="flex items-center gap-1 break-all">
                      <FileText className="h-3 w-3 shrink-0" />
                      llm_jsonl: {logPaths.llmMiningJsonl}
                    </span>
                  )}
                </div>
              )}
            </CardHeader>
            <CardContent className="flex-1 min-h-0 overflow-y-auto bg-slate-950/40 rounded-b-xl border border-slate-800/80 p-4">
              {llmMiningRecent.length === 0 ? (
                <p className="text-sm text-slate-500">暂无 LLM 审计记录；启动自动挖掘或「指定生成」后会出现条目。</p>
              ) : (
                [...llmMiningRecent].reverse().map((rec, i) => (
                  <LlmRecordBlock key={`llm-${i}-${String(rec.logged_at)}-${String(rec.event)}`} rec={rec} />
                ))
              )}
            </CardContent>
          </Card>
        )}

        {/* Real-time Logs (Full Width) */}
        <Card className="glass card-hover animate-fade-in-left lg:col-span-4 h-[400px] flex flex-col">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              实时日志
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 min-h-0 bg-slate-900 rounded-b-xl p-1">
            <div 
              ref={logContainerRef}
              onScroll={handleScroll}
              onMouseLeave={handleMouseLeave}
              className="h-full overflow-y-auto overflow-x-hidden rounded-lg bg-slate-950 p-4 font-mono text-[13px] leading-relaxed space-y-1 border border-slate-800 shadow-inner w-full scroll-smooth whitespace-pre-wrap word-break-all"
            >
              {logs.length === 0 ? (
                <div className="flex h-full items-center justify-center text-slate-500 animate-pulse">
                  等待研究回路日志输出..._
                </div>
              ) : (
                <>
                  {logs.map((log) => (
                    <div key={log.id} className="flex gap-3 items-start hover:bg-slate-800/50 p-0.5 rounded transition-colors group">
                      <span className="text-slate-600 shrink-0 font-semibold select-none w-10">
                        {log.message.includes(']') && log.message.startsWith('[') ? log.message.split(']')[0].slice(1, -3) : 'sys'}
                      </span>
                      <span className={`shrink-0 select-none ${getLogColor(log.level)}`}>{getLogIcon(log.level)}</span>
                      <span className={`${getLogColor(log.level)} opacity-90 break-all inline-block`}>{log.message.includes(']') && log.message.startsWith('[') ? log.message.split(']').slice(1).join(']') : log.message}</span>
                    </div>
                  ))}
                  {/* Anchor for auto-scrolling */}
                  <div ref={logEndRef} className="h-px w-full" />
                </>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
