import React, { useEffect, useMemo, useState } from 'react';
import { BarChart2, FileStack, GitBranch } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/HoverCard';

interface KbFactor {
  run_id: string;
  rank: number;
  formula: string;
  thought_process: string;
  IC: number;
  IR: number;
  tvr: number;
  Score: number;
  PassGates: boolean;
  status: string;
  generation: number;
  created_at: string;
  eval_days: number;
  errors?: string | string[];
  gates_detail?: Record<string, boolean>;
  parquet_path?: string;
  research_path?: string;
  factor_card_path?: string;
  parent_run_ids?: string[];
  live_submitted?: boolean;
  live_test_result?: LiveTestResult;
}

interface LiveTestResult {
  raw: string;
  data: any;
  submitted_at: string;
}

interface AutoAlphaFile {
  name: string;
  path: string;
  relative_path: string;
  kind: string;
  size_bytes: number;
  modified_at: string;
}

interface KnowledgePayload {
  total_tested: number;
  total_passing: number;
  best_score: number;
  updated_at: string;
  pass_rate: number;
  inspiration_stats?: {
    by_source: Array<{
      source: string;
      prompt_count: number;
      tested_count: number;
      passing_count: number;
      pass_rate: number;
      valid_per_prompt: number;
      valid_share: number;
    }>;
    timeline: Array<Record<string, any>>;
    total_passing_attributed: number;
  };
  factors: KbFactor[];
  artifacts: {
    output_files: AutoAlphaFile[];
    research_reports: Array<{
      run_id: string;
      path: string;
      relative_path: string;
      modified_at: string;
      size_bytes: number;
    }>;
  };
}

interface ResearchReport {
  run_id: string;
  formula: string;
  metrics: Record<string, number | boolean>;
  alpha_stats: Record<string, number>;
  factor_card?: FactorCard;
  factor_card_path?: string;
  created_at: string;
}

interface FactorCard {
  run_id: string;
  title: string;
  status: string;
  theme: string;
  thesis: string;
  metrics: Record<string, number>;
  definition?: Record<string, any>;
  distribution?: Record<string, any>;
  histogram?: Array<Record<string, any>>;
  temporal?: Record<string, Array<Record<string, any>>>;
  prediction?: Record<string, any>;
  layering?: Record<string, any>;
  regime?: Array<Record<string, any>>;
  stability?: Record<string, any>;
  redundancy?: Record<string, any>;
  gate_notes: string[];
  diagnostics: Record<string, number | string>;
  risk_notes: string[];
  formula: string;
  created_at: string;
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const payload = await res.json();
  if (!payload.success) throw new Error(payload.error || 'API error');
  return payload.data as T;
}

function formatNumber(value: number, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : '--';
}

function formatMetric(value: any, digits = 4) {
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  const num = Number(value);
  if (!Number.isFinite(num)) return value === undefined || value === null || value === '' ? '--' : String(value);
  return num.toFixed(digits);
}

function formatDateTime(value?: string) {
  if (!value) return '--';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatFileSize(size: number) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function truncate(text: string, length: number) {
  if (!text) return '—';
  return text.length > length ? `${text.slice(0, length)}...` : text;
}

function isSubmitReady(factor: KbFactor) {
  return Boolean(factor.factor_card_path || factor.research_path);
}

function liveResultMetrics(result?: LiveTestResult) {
  const data = result?.data;
  if (Array.isArray(data)) return data[0] || {};
  if (data && typeof data === 'object') return data;
  return {};
}

function statusText(status: string) {
  if (status === 'ok') return '完成';
  if (status === 'invalid') return '公式无效';
  if (status === 'compute_error') return '计算失败';
  if (status === 'screened_out') return '快筛淘汰';
  if (status === 'duplicate') return '重复结构';
  return status || '未知';
}

function failedGateKeys(factor: KbFactor) {
  const detail = factor.gates_detail || {};
  return Object.entries(detail)
    .filter(([, value]) => value === false)
    .map(([key]) => key);
}

function factorFailureReason(factor: KbFactor) {
  if (factor.PassGates) return '通过';
  if (factor.status === 'invalid') return '公式无效';
  if (factor.status === 'compute_error') return '计算失败';
  if (factor.status === 'duplicate') return '重复结构';
  if (factor.status === 'screened_out') return '快筛未达标';
  const failed = failedGateKeys(factor);
  if (failed.length) return failed.map((key) => key.toUpperCase()).join(' / ');
  if ((factor.Score ?? 0) <= 0 && factor.status === 'ok') {
    if ((factor.IC ?? 0) <= 0.6) return 'IC 未过';
    if ((factor.IR ?? 0) <= 2.5) return 'IR 未过';
    if ((factor.tvr ?? 0) >= 400) return 'TVR 过高';
    return 'Score 为 0';
  }
  return statusText(factor.status);
}

function gateTone(reason: string) {
  if (reason.includes('通过')) return 'bg-emerald-100 text-emerald-700';
  if (reason.includes('IC')) return 'bg-sky-100 text-sky-700';
  if (reason.includes('IR')) return 'bg-violet-100 text-violet-700';
  if (reason.includes('TVR') || reason.includes('Turnover')) return 'bg-orange-100 text-orange-700';
  if (reason.includes('快筛')) return 'bg-amber-100 text-amber-700';
  if (reason.includes('重复')) return 'bg-slate-200 text-slate-700';
  return 'bg-red-100 text-red-700';
}

function factorRowBackground(factor: KbFactor, maxScore: number) {
  const reason = factorFailureReason(factor);
  if (factor.status !== 'ok') return 'rgba(239, 68, 68, 0.06)';
  if (!factor.PassGates && reason.includes('IC')) return 'rgba(14, 165, 233, 0.08)';
  if (!factor.PassGates && reason.includes('IR')) return 'rgba(124, 58, 237, 0.08)';
  if (!factor.PassGates && (reason.includes('TVR') || reason.includes('Turnover'))) return 'rgba(249, 115, 22, 0.08)';
  if (!factor.PassGates && reason.includes('Score')) return 'rgba(148, 163, 184, 0.08)';
  if (factor.live_submitted && factor.live_test_result) return 'rgba(20, 184, 166, 0.18)';
  if (factor.Score <= 0 || maxScore <= 0) return 'rgba(148, 163, 184, 0.04)';
  const alpha = Math.min(0.08 + (factor.Score / maxScore) * 0.2, 0.28);
  return `rgba(16, 185, 129, ${alpha})`;
}

const actionButtonClass = "inline-flex h-9 min-w-[4.75rem] items-center justify-center rounded-full border border-border/60 px-3 text-xs font-medium text-foreground transition-colors hover:bg-white";
const submittedButtonClass = "inline-flex h-9 min-w-[4.75rem] items-center justify-center rounded-full bg-emerald-100 px-3 text-xs font-medium text-emerald-700 transition-colors hover:bg-emerald-200";

const Panel = ({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) => (
  <section className="glass rounded-[28px] border border-border/60 p-5">
    <div className="mb-4">
      <div className="text-[11px] uppercase tracking-[0.22em] text-muted-foreground">{title}</div>
      {subtitle ? <div className="mt-2 text-sm text-muted-foreground">{subtitle}</div> : null}
    </div>
    {children}
  </section>
);

const SmallChart = ({
  data,
  type = 'line',
  color = '#0f766e',
  height = 150,
  title,
  description,
}: {
  data?: Array<Record<string, any>>;
  type?: 'line' | 'bar';
  color?: string;
  height?: number;
  title?: string;
  description?: string;
}) => {
  const chartData = (data || []).slice(-80);
  if (!chartData.length) {
    return (
      <div className="rounded-2xl bg-white/80 p-3">
        {title ? <div className="text-xs font-semibold text-foreground">{title}</div> : null}
        {description ? <div className="mt-1 text-[11px] leading-5 text-muted-foreground">{description}</div> : null}
        <div className="mt-2 flex h-[150px] items-center justify-center rounded-2xl bg-slate-50 text-xs text-muted-foreground">暂无序列</div>
      </div>
    );
  }
  return (
    <div className="rounded-2xl bg-white/80 p-3">
      {title ? <div className="text-xs font-semibold text-foreground">{title}</div> : null}
      {description ? <div className="mt-1 text-[11px] leading-5 text-muted-foreground">{description}</div> : null}
      <div style={{ height }} className="mt-2">
        <ResponsiveContainer width="100%" height="100%">
        {type === 'bar' ? (
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
            <XAxis dataKey="x" tick={false} axisLine={false} />
            <YAxis tick={{ fontSize: 10 }} width={38} />
            <Tooltip formatter={(value: any) => formatMetric(value)} labelFormatter={(label) => String(label)} />
            <Bar dataKey="value" fill={color} radius={[4, 4, 0, 0]} />
          </BarChart>
        ) : (
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
            <XAxis dataKey="x" tick={false} axisLine={false} />
            <YAxis tick={{ fontSize: 10 }} width={38} />
            <Tooltip formatter={(value: any) => formatMetric(value)} labelFormatter={(label) => String(label)} />
            <Line type="monotone" dataKey="value" stroke={color} strokeWidth={2} dot={false} />
          </LineChart>
        )}
        </ResponsiveContainer>
      </div>
    </div>
  );
};

const MultiLineChart = ({
  data,
  lines,
  height = 220,
  title,
  description,
}: {
  data?: Array<Record<string, any>>;
  lines: Array<{ key: string; name: string; color: string }>;
  height?: number;
  title: string;
  description?: string;
}) => {
  const chartData = (data || []).slice(-120);
  return (
    <div className="rounded-2xl bg-white/80 p-3">
      <div className="text-xs font-semibold text-foreground">{title}</div>
      {description ? <div className="mt-1 text-[11px] leading-5 text-muted-foreground">{description}</div> : null}
      {!chartData.length ? (
        <div className="mt-2 flex h-[180px] items-center justify-center rounded-2xl bg-slate-50 text-xs text-muted-foreground">暂无序列</div>
      ) : (
        <div style={{ height }} className="mt-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis dataKey="x" tick={false} axisLine={false} />
              <YAxis tick={{ fontSize: 10 }} width={38} />
              <Tooltip formatter={(value: any) => formatMetric(value)} labelFormatter={(label) => String(label)} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {lines.map((line) => (
                <Line key={line.key} type="monotone" dataKey={line.key} name={line.name} stroke={line.color} strokeWidth={2} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

const StatGrid = ({ items }: { items: Array<[string, any]> }) => (
  <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
    {items.map(([label, value]) => (
      <div key={label} className="rounded-2xl bg-white/85 p-3">
        <div className="text-[11px] text-muted-foreground">{label}</div>
        <div className="mt-1 font-mono text-sm font-semibold text-foreground">{formatMetric(value)}</div>
      </div>
    ))}
  </div>
);

const CardSection = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <div className="rounded-3xl border border-border/50 bg-slate-50/80 p-4">
    <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-foreground">
      <BarChart2 className="h-4 w-4 text-emerald-600" />
      {title}
    </div>
    {children}
  </div>
);

const sourceLabel = (source: string) => {
  if (source === 'arxiv') return 'ArXiv';
  if (source === 'llm') return 'LLM';
  if (source === 'future') return 'Future';
  return source;
};

const InspirationStatsCharts = ({ stats }: { stats?: KnowledgePayload['inspiration_stats'] }) => {
  const bySource = stats?.by_source || [];
  const timeline = stats?.timeline || [];
  const sourceColors: Record<string, string> = {
    arxiv: '#2563eb',
    llm: '#7c3aed',
    future: '#f97316',
  };

  if (!bySource.length && !timeline.length) {
    return <div className="rounded-3xl bg-white/80 p-5 text-sm text-muted-foreground">暂无灵感源统计。抓取或同步灵感后，这里会显示来源转化率。</div>;
  }

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <div className="rounded-3xl border border-border/50 bg-white/90 p-4">
        <div className="mb-3">
          <div className="text-sm font-medium text-foreground">灵感源转化</div>
          <div className="mt-1 text-xs text-muted-foreground">左轴为灵感次数 / passing 因子数，右轴为通过率。</div>
        </div>
        <div className="h-[280px]">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={bySource}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis dataKey="source" tickFormatter={sourceLabel} tick={{ fontSize: 11 }} />
              <YAxis yAxisId="left" tick={{ fontSize: 11 }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} tickFormatter={(value) => `${value}%`} />
              <Tooltip formatter={(value: any) => formatMetric(value)} labelFormatter={(label) => sourceLabel(String(label))} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar yAxisId="left" dataKey="prompt_count" name="灵感次数" fill="#94a3b8" radius={[4, 4, 0, 0]} />
              <Bar yAxisId="left" dataKey="passing_count" name="Passing 因子" fill="#10b981" radius={[4, 4, 0, 0]} />
              <Line yAxisId="right" type="monotone" dataKey="pass_rate" name="通过率 %" stroke="#ef4444" strokeWidth={2} dot />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="rounded-3xl border border-border/50 bg-white/90 p-4">
        <div className="mb-3">
          <div className="text-sm font-medium text-foreground">Prompt 有效产出</div>
          <div className="mt-1 text-xs text-muted-foreground">左轴为每个 Prompt 平均有效因子数，右轴为该来源占全部有效因子的比例。</div>
        </div>
        <div className="h-[280px]">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={bySource}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis dataKey="source" tickFormatter={sourceLabel} tick={{ fontSize: 11 }} />
              <YAxis yAxisId="left" tick={{ fontSize: 11 }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} tickFormatter={(value) => `${value}%`} />
              <Tooltip formatter={(value: any) => formatMetric(value)} labelFormatter={(label) => sourceLabel(String(label))} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar yAxisId="left" dataKey="valid_per_prompt" name="有效因子 / Prompt" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
              <Line yAxisId="right" type="monotone" dataKey="valid_share" name="有效因子占比 %" stroke="#7c3aed" strokeWidth={2} dot />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="rounded-3xl border border-border/50 bg-white/90 p-4 xl:col-span-2">
        <div className="mb-3">
          <div className="text-sm font-medium text-foreground">通过率随研究记录变化</div>
          <div className="mt-1 text-xs text-muted-foreground">每条线表示该来源累计通过率，横轴为因子记录序号。</div>
        </div>
        <div className="h-[240px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={timeline}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={(value) => `${value}%`} />
              <Tooltip formatter={(value: any) => `${formatMetric(value)}%`} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {(['arxiv', 'llm', 'future'] as const).map((source) => (
                <Line key={source} type="monotone" dataKey={`${source}_pass_rate`} name={`${sourceLabel(source)} 通过率`} stroke={sourceColors[source]} strokeWidth={2} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

const ResearchModal = ({ runId, onClose }: { runId: string; onClose: () => void }) => {
  const [report, setReport] = useState<ResearchReport | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let mounted = true;
    fetchJson<{ report: ResearchReport }>(`/api/autoalpha/research/${runId}`)
      .then((data) => {
        if (mounted) setReport(data.report);
      })
      .catch((err: Error) => {
        if (mounted) setError(err.message);
      });
    return () => {
      mounted = false;
    };
  }, [runId]);

  const card = report?.factor_card;
  const cardMetrics = card?.metrics || {};
  const diagnostics = card?.diagnostics || {};

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4" onClick={onClose}>
      <div className="max-h-[88vh] w-full max-w-4xl overflow-y-auto rounded-[28px] border border-border/50 bg-white p-6 shadow-2xl" onClick={(event) => event.stopPropagation()}>
        <div className="mb-5 flex items-center justify-between gap-3">
          <div>
            <div className="text-[11px] uppercase tracking-[0.22em] text-muted-foreground">Factor Card</div>
            <div className="mt-2 text-xl font-semibold text-foreground">{runId}</div>
          </div>
          <button onClick={onClose} className="rounded-full border border-border/60 px-3 py-1 text-sm text-muted-foreground transition-colors hover:text-foreground">
            关闭
          </button>
        </div>
        {error ? <div className="rounded-2xl bg-red-50 p-4 text-sm text-red-600">{error}</div> : null}
        {!report && !error ? <div className="text-sm text-muted-foreground">Factor Card 加载中...</div> : null}
        {report ? (
          <div className="space-y-4">
            {card && card.status === 'PASS' ? (
              <div className="space-y-4 rounded-3xl border border-emerald-200 bg-emerald-50/60 p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="text-[11px] uppercase tracking-[0.22em] text-emerald-700">Factor Card</div>
                    <div className="mt-2 break-all text-lg font-semibold text-slate-950">{card.title || card.run_id}</div>
                    <div className="mt-1 text-sm text-emerald-800">{card.theme}</div>
                  </div>
                  <div className={`rounded-full px-3 py-1 text-xs font-semibold ${card.status === 'PASS' ? 'bg-emerald-600 text-white' : 'bg-slate-200 text-slate-700'}`}>
                    {card.status}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  {['IC', 'IR', 'tvr', 'Score'].map((key) => (
                    <div key={key} className="rounded-2xl bg-white/90 p-3">
                      <div className="text-[11px] text-muted-foreground">{key}</div>
                      <div className="mt-2 font-mono text-lg font-semibold text-foreground">{formatMetric(cardMetrics[key])}</div>
                    </div>
                  ))}
                </div>
                <CardSection title="1. 因子定义">
                  <div className="space-y-3 text-sm leading-6 text-slate-700">
                    <pre className="whitespace-pre-wrap break-all rounded-2xl bg-white/90 p-3 font-mono text-xs">{card.definition?.formula || card.formula}</pre>
                    <div className="grid gap-2 md:grid-cols-2">
                      <div>输入字段：{(card.definition?.inputs || []).join(', ') || '--'}</div>
                      <div>更新频率：{card.definition?.update_frequency || '15-minute'}</div>
                      <div>后处理：{card.definition?.postprocess || '--'}</div>
                    </div>
                    <div>{card.thesis}</div>
                  </div>
                </CardSection>
                <CardSection title="2. 历史分布">
                  <div className="grid gap-3 lg:grid-cols-[1fr_1.2fr]">
                    <SmallChart
                      title="因子取值直方图"
                      description="横轴是因子值区间，柱越高表示该区间出现越多；过度偏斜或极端值过多会带来不稳定。"
                      data={(card.histogram || []).map((row) => ({ x: row.bin, value: row.value }))}
                      type="bar"
                      color="#7c3aed"
                    />
                    <StatGrid items={[
                      ['P1', card.distribution?.p1], ['P5', card.distribution?.p5], ['P50', card.distribution?.p50], ['P95', card.distribution?.p95],
                      ['P99', card.distribution?.p99], ['Mean', card.distribution?.mean], ['Std', card.distribution?.std], ['Skew', card.distribution?.skew],
                      ['Kurt', card.distribution?.kurt], ['Missing', card.distribution?.missing_rate], ['Extreme', card.distribution?.extreme_share], ['InBounds', diagnostics.pct_in_bounds],
                    ]} />
                  </div>
                </CardSection>
                <CardSection title="3. 时序演变">
                  <div className="grid gap-3 md:grid-cols-3">
                    <SmallChart title="日均因子暴露" description="看因子整体方向是否长期漂移；持续单边漂移通常需要检查归一化。" data={card.temporal?.daily_mean} color="#0f766e" />
                    <SmallChart title="日内截面离散度" description="数值越高，股票间区分度越强；塌缩到低位代表信号信息不足。" data={card.temporal?.daily_std} color="#2563eb" />
                    <SmallChart title="有效覆盖率" description="接近 1 表示大部分标的有因子值；突然下坠要检查缺失或限制过滤。" data={card.temporal?.coverage} color="#f97316" />
                  </div>
                </CardSection>
                <CardSection title="4. 预测力">
                  <div className="grid gap-3 lg:grid-cols-[1fr_1.2fr]">
                    <StatGrid items={[
                      ['IC Mean', card.prediction?.ic_mean ?? cardMetrics.IC], ['ICIR', card.prediction?.icir ?? cardMetrics.IR],
                      ['Rank IC', card.prediction?.rank_ic], ['Half Life', diagnostics.ic_half_life],
                    ]} />
                    <SmallChart title="20 日滚动 IC" description="线上越稳定高于 0，说明近期预测方向越稳定；穿 0 表示信号可能换 regime。" data={card.prediction?.rolling_ic} color="#0f766e" />
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-4">
                    {(card.prediction?.horizon_ic || []).map((row: any) => (
                      <div key={row.horizon} className="rounded-2xl bg-white/85 p-3 text-xs">
                        <div className="text-muted-foreground">{row.horizon}</div>
                        <div className="mt-1 font-mono font-semibold">{formatMetric(row.ic)}</div>
                      </div>
                    ))}
                  </div>
                </CardSection>
                <CardSection title="5. 表现分层图">
                  <div className="grid gap-3 lg:grid-cols-[1fr_1.2fr]">
                    <SmallChart title="因子分位收益" description="Q10 高于 Q1 表示因子排序和未来收益同向；单调性越强，信号越可信。" data={(card.layering?.decile_returns_bps || []).map((row: any) => ({ x: `Q${row.bucket}`, value: row.value }))} type="bar" color="#0ea5e9" />
                    <div>
                      <StatGrid items={[['Top-Bottom bps', card.layering?.top_minus_bottom_bps], ['Score', cardMetrics.Score], ['TVR', cardMetrics.tvr], ['Days', cardMetrics.nd]]} />
                      <div className="mt-3"><SmallChart title="多空分层累计收益" description="累计线上行代表高分位组合持续跑赢低分位组合。" data={card.layering?.cumulative_top_minus_bottom} color="#16a34a" height={120} /></div>
                    </div>
                  </div>
                </CardSection>
                <CardSection title="6. 什么时候表现好">
                  <MultiLineChart
                    title="IC 与市场状态同图"
                    description="全部序列已标准化到同一坐标：IC 上行且价格/波动/成交额处在对应位置时，就是这个因子更适合的市场环境。"
                    data={card.temporal?.market_state}
                    lines={[
                      { key: 'ic', name: 'IC', color: '#0f766e' },
                      { key: 'price', name: '价格指数', color: '#2563eb' },
                      { key: 'volatility', name: '波动', color: '#f97316' },
                      { key: 'liquidity', name: '成交额', color: '#7c3aed' },
                    ]}
                  />
                  <div className="grid gap-2 md:grid-cols-3">
                    {(card.regime || []).map((row) => (
                      <div key={String(row.regime)} className="rounded-2xl bg-white/85 p-3">
                        <div className="text-xs text-muted-foreground">{String(row.regime)}</div>
                        <div className="mt-1 font-mono text-sm font-semibold">IC {formatMetric(row.ic)}</div>
                        <div className="text-[11px] text-muted-foreground">{row.days} days</div>
                      </div>
                    ))}
                  </div>
                </CardSection>
                <CardSection title="7. 稳定性">
                  <div className="grid gap-3 lg:grid-cols-[1fr_1.2fr]">
                    <SmallChart title="月度 IC" description="按自然月统计全样本 IC，不再拆 Train/Val/Test；越少跨月翻负越稳定。" data={card.stability?.monthly_ic} color="#9333ea" />
                    <StatGrid items={[
                      ['全样本 IC', card.stability?.full_sample_ic],
                      ['月度为正占比', card.stability?.positive_month_ratio],
                      ['最差月 IC', card.stability?.worst_month_ic],
                      ['最好月 IC', card.stability?.best_month_ic],
                      ['Clipped IC', card.stability?.clipped_ic],
                    ]} />
                  </div>
                </CardSection>
                <CardSection title="8. 相关性与冗余">
                  <StatGrid items={[
                    ['Token Overlap', card.redundancy?.max_formula_token_overlap],
                    ['Target Proxy', cardMetrics.IC ? Number(cardMetrics.IC) / 100 : undefined],
                    ['Family', card.redundancy?.family],
                    ['Nearest', card.redundancy?.nearest_factor || '--'],
                  ]} />
                  <div className="mt-3 rounded-2xl bg-white/80 p-3 text-xs leading-5 text-slate-700">
                    {(card.gate_notes || []).map((note) => <div key={note}>{note}</div>)}
                    {(card.risk_notes || []).map((note) => <div key={note}>{note}</div>)}
                  </div>
                </CardSection>
              </div>
            ) : null}
            <div className="rounded-3xl border border-border/50 bg-slate-50 p-4">
              <div className="text-xs text-muted-foreground">公式</div>
              <pre className="mt-2 whitespace-pre-wrap break-all text-xs text-slate-700">{report.formula}</pre>
            </div>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
              {['IC', 'IR', 'Turnover', 'Score', 'PassGates'].map((key) => (
                <div key={key} className="rounded-2xl border border-border/50 bg-white p-3">
                  <div className="text-[11px] text-muted-foreground">{key}</div>
                  <div className="mt-2 break-words font-mono font-semibold text-foreground">{formatMetric(report.metrics?.[key])}</div>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
};

const LiveResultModal = ({
  factor,
  value,
  error,
  saving,
  onChange,
  onSave,
  onClose,
}: {
  factor: KbFactor;
  value: string;
  error: string;
  saving: boolean;
  onChange: (value: string) => void;
  onSave: () => void;
  onClose: () => void;
}) => (
  <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4" onClick={onClose}>
    <div className="w-full max-w-3xl rounded-[28px] border border-border/50 bg-white p-6 shadow-2xl" onClick={(event) => event.stopPropagation()}>
      <div className="mb-5 flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[11px] uppercase tracking-[0.22em] text-muted-foreground">Lab Test 结果</div>
          <div className="mt-2 break-all text-lg font-semibold text-foreground">{factor.run_id}</div>
        </div>
        <button onClick={onClose} className="rounded-full border border-border/60 px-3 py-1 text-sm text-muted-foreground transition-colors hover:text-foreground">
          关闭
        </button>
      </div>
      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="[{'IC': 0.8086, 'IR': 4.2816, 'tvr': 499.092, ...}]"
        className="h-64 w-full rounded-2xl border border-border/60 bg-slate-50 px-4 py-3 font-mono text-xs leading-6 outline-none focus:border-slate-400"
      />
      {error ? <div className="mt-3 rounded-2xl bg-red-50 px-4 py-3 text-sm text-red-600">{error}</div> : null}
      <div className="mt-4 flex justify-end gap-3">
        <button onClick={onClose} className="rounded-2xl border border-border/60 px-4 py-2 text-sm text-foreground transition-colors hover:bg-slate-50">
          取消
        </button>
        <button onClick={onSave} disabled={saving || !value.trim()} className="rounded-2xl bg-slate-950 px-4 py-2 text-sm text-white transition-colors hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50">
          {saving ? '保存中...' : '保存 Lab Test'}
        </button>
      </div>
    </div>
  </div>
);

export const AutoAlphaRecordsPage: React.FC = () => {
  const [knowledge, setKnowledge] = useState<KnowledgePayload | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [liveResultFactor, setLiveResultFactor] = useState<KbFactor | null>(null);
  const [liveResultText, setLiveResultText] = useState('');
  const [liveResultError, setLiveResultError] = useState('');
  const [liveResultSaving, setLiveResultSaving] = useState(false);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        const data = await fetchJson<KnowledgePayload>('/api/autoalpha/knowledge');
        if (mounted) setKnowledge(data);
      } catch {
        // Keep the records page usable even when a poll fails.
      }
    };
    load();
    const timer = window.setInterval(load, 5000);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, []);

  const factors = knowledge?.factors ?? [];
  const outputFiles = knowledge?.artifacts.output_files ?? [];
  const researchReports = knowledge?.artifacts.research_reports ?? [];
  const maxScore = Math.max(...factors.map((factor) => factor.Score), 1);
  const lineageLanes = useMemo(
    () =>
      Array.from(
        factors.reduce((acc, factor) => {
          const list = acc.get(factor.generation) || [];
          list.push(factor);
          acc.set(factor.generation, list);
          return acc;
        }, new Map<number, KbFactor[]>())
      )
        .sort((a, b) => a[0] - b[0])
        .map(([generation, laneFactors]) => ({
          generation,
          factors: laneFactors.sort((a, b) => b.Score - a.Score).slice(0, 8),
        })),
    [factors]
  );

  const openLiveResultModal = (factor: KbFactor) => {
    setLiveResultFactor(factor);
    setLiveResultText(factor.live_test_result?.raw || '');
    setLiveResultError('');
  };

  const saveLiveResult = async () => {
    if (!liveResultFactor) return;
    setLiveResultSaving(true);
    setLiveResultError('');
    try {
      const data = await fetchJson<{ run_id: string; live_test_result: LiveTestResult; live_submitted: boolean }>(
        `/api/autoalpha/factors/${liveResultFactor.run_id}/live-result`,
        {
          method: 'POST',
          body: JSON.stringify({ result_text: liveResultText }),
        }
      );
      setKnowledge((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          factors: prev.factors.map((factor) =>
            factor.run_id === data.run_id
              ? { ...factor, live_test_result: data.live_test_result, live_submitted: data.live_submitted }
              : factor
          ),
        };
      });
      setLiveResultFactor(null);
      setLiveResultText('');
    } catch (error: any) {
      setLiveResultError(error.message || '保存失败');
    } finally {
      setLiveResultSaving(false);
    }
  };

  return (
    <div className="space-y-6 pb-10">
      <Panel title="AutoAlpha 记录库" subtitle="Generation 演进、产出文件、因子卡片和知识库因子表">
        <div className="grid gap-4 md:grid-cols-4">
          <div className="rounded-3xl bg-white/80 p-4">
            <div className="text-xs text-muted-foreground">已测试</div>
            <div className="mt-2 text-2xl font-semibold">{knowledge?.total_tested ?? 0}</div>
          </div>
          <div className="rounded-3xl bg-emerald-50 p-4">
            <div className="text-xs text-muted-foreground">通过 Gate</div>
            <div className="mt-2 text-2xl font-semibold">{knowledge?.total_passing ?? 0}</div>
          </div>
          <div className="rounded-3xl bg-sky-50 p-4">
            <div className="text-xs text-muted-foreground">最佳 Score</div>
            <div className="mt-2 text-2xl font-semibold">{formatNumber(knowledge?.best_score ?? 0, 2)}</div>
          </div>
          <div className="rounded-3xl bg-violet-50 p-4">
            <div className="text-xs text-muted-foreground">最近更新</div>
            <div className="mt-2 text-lg font-semibold">{formatDateTime(knowledge?.updated_at)}</div>
          </div>
        </div>
      </Panel>

      <Panel title="灵感源转化分析" subtitle="按 ArXiv、LLM 和 Future Markdown 追踪灵感次数、passing 因子数、通过率与有效因子贡献。">
        <InspirationStatsCharts stats={knowledge?.inspiration_stats} />
      </Panel>

      <Panel title="进化 Generation 记录">
        <div className="max-h-[620px] overflow-auto rounded-3xl border border-border/40 bg-white/70 p-3">
          <div className="flex min-w-max items-start gap-4 pb-2">
            {lineageLanes.length === 0 ? (
              <div className="text-sm text-muted-foreground">还没有代际数据。</div>
            ) : (
              lineageLanes.map((lane, index) => (
                <div key={lane.generation} className="flex items-start gap-4">
                  <div className="w-[280px]">
                    <div className="mb-3 flex items-center gap-2 text-sm font-medium text-foreground">
                      <GitBranch className="h-4 w-4 text-emerald-500" />
                      Generation {lane.generation}
                    </div>
                    <div className="space-y-3">
                      {lane.factors.map((factor) => (
                        <div key={factor.run_id} className="rounded-2xl border border-border/50 bg-white p-3 shadow-sm">
                          <div className="flex items-center justify-between gap-3">
                            <div className="min-w-0">
                              {factor.factor_card_path || factor.research_path ? (
                                <button onClick={() => setSelectedRunId(factor.run_id)} className="block max-w-full truncate font-mono text-xs font-medium text-sky-700 underline decoration-sky-400 underline-offset-4 hover:text-sky-900">
                                  {factor.run_id}
                                </button>
                              ) : (
                                <div className="truncate font-mono text-xs text-foreground">{factor.run_id}</div>
                              )}
                              <div className="mt-1 text-[11px] text-muted-foreground">Score {formatNumber(factor.Score, 2)} · IC {formatNumber(factor.IC, 3)}</div>
                            </div>
                            <div className={`rounded-full px-2 py-1 text-[10px] ${factor.PassGates ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-600'}`}>
                              {factor.PassGates ? 'PASS' : 'TEST'}
                            </div>
                          </div>
                          <div className="mt-2 break-words text-xs leading-6 text-slate-700">{truncate(factor.formula, 110)}</div>
                          <div className="mt-3 flex flex-wrap gap-2">
                            {(factor.parent_run_ids || []).slice(0, 3).map((parent) => (
                              <span key={`${factor.run_id}-${parent}`} className="rounded-full bg-slate-100 px-2 py-1 text-[10px] text-slate-600">
                                {truncate(parent, 14)}
                              </span>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  {index < lineageLanes.length - 1 ? <div className="mt-24 text-slate-300">-&gt;</div> : null}
                </div>
              ))
            )}
          </div>
        </div>
      </Panel>

      <Panel title="文件与因子卡片留存">
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-3xl border border-border/50 bg-white/90 p-4">
            <div className="mb-3 flex items-center gap-2 text-sm font-medium text-foreground">
              <FileStack className="h-4 w-4 text-sky-500" />
              最近输出文件
            </div>
            <div className="max-h-[460px] space-y-3 overflow-y-auto pr-2">
              {outputFiles.length === 0 ? (
                <div className="text-sm text-muted-foreground">暂时还没有输出文件。</div>
              ) : (
                outputFiles.map((file) => (
                  <div key={file.path} className="rounded-2xl border border-border/50 bg-slate-50 p-3">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0 flex-1">
                        <div className="break-all font-mono text-xs text-foreground">{file.name}</div>
                        <div className="mt-1 break-all text-[11px] leading-5 text-muted-foreground">{file.relative_path}</div>
                      </div>
                      <div className="shrink-0 rounded-full bg-slate-200 px-2 py-1 text-[10px] uppercase text-slate-700">{file.kind}</div>
                    </div>
                    <div className="mt-3 flex items-center justify-between gap-3 text-[11px] text-muted-foreground">
                      <span>{formatFileSize(file.size_bytes)}</span>
                      <span>{formatDateTime(file.modified_at)}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="rounded-3xl border border-border/50 bg-white/90 p-4">
            <div className="mb-3 text-sm font-medium text-foreground">因子卡片索引</div>
            <div className="max-h-[460px] space-y-3 overflow-y-auto pr-2">
              {researchReports.length === 0 ? (
                <div className="text-sm text-muted-foreground">当前还没有因子卡片文件。</div>
              ) : (
                researchReports.map((report) => (
                  <div key={report.path} className="rounded-2xl border border-border/50 bg-slate-50 p-3">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0 flex-1">
                        <div className="break-all font-medium text-foreground">{report.run_id}</div>
                        <div className="mt-1 break-all text-[11px] leading-5 text-muted-foreground">{report.relative_path}</div>
                      </div>
                      <button onClick={() => setSelectedRunId(report.run_id)} className="shrink-0 rounded-full border border-border/60 px-3 py-1 text-xs text-foreground transition-colors hover:bg-white">
                        打开
                      </button>
                    </div>
                    <div className="mt-3 flex items-center justify-between gap-3 text-[11px] text-muted-foreground">
                      <span>{formatFileSize(report.size_bytes)}</span>
                      <span>{formatDateTime(report.modified_at)}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </Panel>

      <Panel title="知识库因子表" subtitle="Score 为 0 或未通过 Gate 的因子会在 Status/Gate 列注明原因，不同失败原因会用不同底色标记整行。">
        <div className="max-h-[720px] overflow-auto rounded-3xl border border-border/40 bg-white/70">
          <table className="min-w-[1500px] table-fixed text-sm">
            <thead className="sticky top-0 z-10 bg-white">
              <tr className="border-b border-border/50 text-left text-xs uppercase tracking-[0.18em] text-muted-foreground">
                <th className="w-16 px-3 py-3">Rank</th>
                <th className="w-36 px-3 py-3">Run ID</th>
                <th className="w-20 px-3 py-3 text-right">Score</th>
                <th className="w-16 px-3 py-3 text-right">IC</th>
                <th className="w-16 px-3 py-3 text-right">IR</th>
                <th className="w-16 px-3 py-3 text-right">TVR</th>
                <th className="w-16 px-3 py-3 text-center">Days</th>
                <th className="w-16 px-3 py-3 text-center">Gen</th>
                <th className="w-[30rem] px-3 py-3">Formula</th>
                <th className="w-[22rem] px-3 py-3">Thought</th>
                <th className="w-48 px-3 py-3">Status/Gate</th>
                <th className="w-24 px-3 py-3 text-center">Lab Test</th>
              </tr>
            </thead>
            <tbody>
              {factors.length === 0 ? (
                <tr>
                  <td colSpan={12} className="px-3 py-12 text-center text-sm text-muted-foreground">
                    还没有因子记录。启动循环后，这里会持续刷新。
                  </td>
                </tr>
              ) : (
                factors.map((factor) => {
                  const reason = factorFailureReason(factor);
                  return (
                    <tr key={factor.run_id} className="border-b border-border/20 transition-colors" style={{ background: factorRowBackground(factor, maxScore) }}>
                      <td className="px-3 py-3 font-semibold text-foreground">#{factor.rank}</td>
                      <td className="px-3 py-3 align-top">
                        {isSubmitReady(factor) ? (
                          <button onClick={() => setSelectedRunId(factor.run_id)} className="break-all text-left font-mono text-xs font-medium text-sky-700 underline decoration-sky-400 underline-offset-4 hover:text-sky-900">
                            {factor.run_id}
                          </button>
                        ) : (
                          <div className="break-all font-mono text-xs text-foreground">{factor.run_id}</div>
                        )}
                        <div className="mt-1 text-[11px] text-muted-foreground">{formatDateTime(factor.created_at)}</div>
                      </td>
                      <td className="px-3 py-3 text-right font-semibold text-foreground">{formatNumber(factor.Score, 2)}</td>
                      <td className="px-3 py-3 text-right font-mono text-foreground">{formatNumber(factor.IC, 3)}</td>
                      <td className="px-3 py-3 text-right font-mono text-foreground">{formatNumber(factor.IR, 3)}</td>
                      <td className="px-3 py-3 text-right font-mono text-foreground">{formatNumber(factor.tvr, 0)}</td>
                      <td className="px-3 py-3 text-center text-foreground">{factor.eval_days || '-'}</td>
                      <td className="px-3 py-3 text-center text-foreground">{factor.generation}</td>
                      <td className="px-3 py-3 align-top">
                        <HoverCard>
                          <HoverCardTrigger asChild>
                            <button className="w-full whitespace-pre-wrap break-all text-left font-mono text-xs leading-6 text-slate-700 hover:text-slate-950">{truncate(factor.formula, 160)}</button>
                          </HoverCardTrigger>
                          <HoverCardContent align="start" className="w-[620px] border border-slate-200 bg-white shadow-2xl backdrop-blur-none">
                            <div className="space-y-2">
                              <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Formula</div>
                              <pre className="whitespace-pre-wrap break-all bg-white font-mono text-xs leading-6 text-foreground">{factor.formula}</pre>
                              {(factor.parent_run_ids || []).length ? (
                                <div className="text-[11px] text-muted-foreground">Parents: {(factor.parent_run_ids || []).join(', ')}</div>
                              ) : null}
                              {factor.parquet_path ? <div className="break-all text-[11px] text-muted-foreground">Parquet: {factor.parquet_path}</div> : null}
                            </div>
                          </HoverCardContent>
                        </HoverCard>
                      </td>
                      <td className="px-3 py-3 align-top">
                        <HoverCard>
                          <HoverCardTrigger asChild>
                            <button className="w-full whitespace-pre-wrap break-words text-left text-xs leading-6 text-slate-700 hover:text-slate-950">
                              {truncate(factor.thought_process || String(factor.errors || '') || '暂无说明', 120)}
                            </button>
                          </HoverCardTrigger>
                          <HoverCardContent align="start" className="w-[460px] border border-slate-200 bg-white shadow-2xl backdrop-blur-none">
                            <div className="space-y-3 text-xs leading-6">
                              <div>
                                <div className="uppercase tracking-[0.18em] text-muted-foreground">Thought Process</div>
                                <div className="mt-2 text-foreground">{factor.thought_process || '暂无 thought process'}</div>
                              </div>
                              {factor.errors ? (
                                <div>
                                  <div className="uppercase tracking-[0.18em] text-muted-foreground">Error</div>
                                  <div className="mt-2 text-red-600">{Array.isArray(factor.errors) ? factor.errors.join('; ') : factor.errors}</div>
                                </div>
                              ) : null}
                            </div>
                          </HoverCardContent>
                        </HoverCard>
                      </td>
                      <td className="px-3 py-3 align-top">
                        <div className={`inline-flex max-w-full rounded-full px-3 py-1 text-xs font-medium ${gateTone(reason)}`}>
                          <span className="break-words">{reason}</span>
                        </div>
                        <div className="mt-1 text-[11px] text-muted-foreground">{statusText(factor.status)}</div>
                      </td>
                      <td className="px-3 py-3 align-top">
                        {factor.live_submitted && factor.live_test_result ? (
                          <HoverCard>
                            <HoverCardTrigger asChild>
                              <button onClick={() => openLiveResultModal(factor)} className={submittedButtonClass}>
                                已提交
                              </button>
                            </HoverCardTrigger>
                            <HoverCardContent align="end" className="w-[520px] border border-slate-200 bg-white shadow-2xl backdrop-blur-none">
                              <div className="space-y-3 text-xs">
                                <div className="flex items-center justify-between gap-3">
                                  <div className="uppercase tracking-[0.18em] text-muted-foreground">Lab Test</div>
                                  <div className="text-muted-foreground">{formatDateTime(factor.live_test_result.submitted_at)}</div>
                                </div>
                                <div className="grid grid-cols-4 gap-2">
                                  {['IC', 'IR', 'tvr', 'score', 'cover_all', 'nd', 'max', 'min'].map((key) => {
                                    const metrics = liveResultMetrics(factor.live_test_result);
                                    return (
                                      <div key={key} className="rounded-xl bg-slate-50 p-2">
                                        <div className="text-[10px] text-muted-foreground">{key}</div>
                                        <div className="mt-1 break-words font-mono text-foreground">{formatMetric(metrics[key])}</div>
                                      </div>
                                    );
                                  })}
                                </div>
                                <pre className="max-h-44 overflow-auto whitespace-pre-wrap break-all rounded-xl bg-slate-50 p-3 font-mono text-[11px] leading-5 text-slate-700">{factor.live_test_result.raw}</pre>
                              </div>
                            </HoverCardContent>
                          </HoverCard>
                        ) : (
                          <button onClick={() => openLiveResultModal(factor)} className={actionButtonClass}>
                            填入
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </Panel>

      {selectedRunId ? <ResearchModal runId={selectedRunId} onClose={() => setSelectedRunId(null)} /> : null}
      {liveResultFactor ? (
        <LiveResultModal
          factor={liveResultFactor}
          value={liveResultText}
          error={liveResultError}
          saving={liveResultSaving}
          onChange={setLiveResultText}
          onSave={saveLiveResult}
          onClose={() => setLiveResultFactor(null)}
        />
      ) : null}
    </div>
  );
};
