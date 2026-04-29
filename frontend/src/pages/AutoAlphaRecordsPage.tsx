import React, { useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import { BarChart2, Download, FileStack, FileText, GitBranch } from 'lucide-react';
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
  screen_fail_reason?: string;
  screen_fail_details?: ScreenFailDetail[];
  screening?: Record<string, any>;
  gates_detail?: Record<string, boolean>;
  parquet_path?: string;
  submit_path?: string;
  submit_metadata_path?: string;
  research_path?: string;
  factor_card_path?: string;
  download_available?: boolean;
  parent_run_ids?: string[];
  oss_2024?: {
    start?: string;
    end?: string;
    days?: number;
    metrics?: Record<string, any>;
    used_for_feedback?: boolean;
  };
  full_2022_2024?: {
    start?: string;
    end?: string;
    days?: number;
    metrics?: Record<string, any>;
    used_for_discovery?: boolean;
  };
  period_metrics?: PeriodMetric[];
  oos_comparison?: OosComparison;
}

interface PeriodMetric {
  period: string;
  label: string;
  start?: string;
  end?: string;
  days?: number;
  IC?: number;
  IR?: number;
  tvr?: number;
  Turnover?: number;
  Score?: number;
  PassGates?: boolean;
  used_for_discovery?: boolean;
}

interface OosComparison {
  IC_delta?: number;
  Score_delta?: number;
  IR_delta?: number;
  tvr_delta?: number;
  IC_retention?: number;
  Score_retention?: number;
  trend?: string;
  used_for_feedback?: boolean;
}

interface ScreenFailDetail {
  key: string;
  value?: number | string;
  threshold?: number | string;
  direction?: string;
  message?: string;
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
    source_mix_note?: string;
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
  factor_correlations?: FactorCorrelationPayload;
  generation_experiences?: GenerationExperience[];
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

interface FactorCorrelationPayload {
  labels: string[];
  run_ids?: string[];
  matrix: (number | null)[][];
  updated_at?: string;
  meta?: {
    heatmap_stale?: boolean;
    trend_stale?: boolean;
    expected_heatmap_count?: number;
    expected_trend_count?: number;
    actual_heatmap_count?: number;
    actual_trend_count?: number;
    refreshing?: boolean;
  };
  basis?: string;
  trend_basis?: string;
  trend_rows?: Array<{
    index: number;
    label?: string;
    run_id: string;
    created_at?: string;
    generation?: number;
    tested_index?: number;
    score?: number;
    ic?: number;
    avg_corr: number;
    max_corr: number;
    min_corr: number;
    pair_count?: number;
    basis?: string;
  }>;
}

interface GenerationExperience {
  generation: number;
  created_at: string;
  path: string;
  relative_path: string;
  summary: string;
  markdown?: string;
  stats?: {
    total: number;
    passing: number;
    best_score: number;
    failure_counts: Record<string, number>;
  };
}

interface ResearchReport {
  run_id: string;
  formula: string;
  metrics: Record<string, number | boolean>;
  alpha_stats?: Record<string, any>;
  daily_ic?: Record<string, number>;
  n_daily_ic?: number;
  factor_card?: FactorCard;
  factor_card_path?: string;
  created_at: string;
}

interface FactorCard {
  run_id: string;
  title: string;
  status: string;
  theme?: string;
  thesis?: string;
  metrics: Record<string, number>;
  definition?: Record<string, any>;
  distribution?: Record<string, any>;
  histogram?: Array<Record<string, any>>;
  temporal?: Record<string, Array<Record<string, any>>>;
  prediction?: Record<string, any>;
  monthly_ic?: Record<string, any>;
  regime?: Array<Record<string, any>>;
  redundancy?: Record<string, any>;
  gate_notes?: string[];
  diagnostics?: Record<string, number | string>;
  risk_notes?: string[];
  formula: string;
  created_at: string;
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const text = await res.text();
  let payload: any = {};
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    const preview = text.slice(0, 140).replace(/\s+/g, ' ').trim();
    throw new Error(`API 返回非 JSON (${res.status}): ${preview || res.statusText}`);
  }
  if (!res.ok) {
    throw new Error(payload.error || `API HTTP ${res.status}`);
  }
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

function firstMetric(source: Record<string, any> | undefined, keys: string[]) {
  if (!source) return undefined;
  for (const key of keys) {
    const value = source[key];
    if (value !== undefined && value !== null && value !== '') return value;
  }
  return undefined;
}

function rollingDailyIcSeries(dailyIc?: Record<string, number>) {
  const rows = Object.entries(dailyIc || {})
    .map(([x, value]) => ({ x: compactDateLabel(x), rawX: x, value: Number(value) }))
    .filter((row) => Number.isFinite(row.value))
    .sort((a, b) => String(a.rawX).localeCompare(String(b.rawX)));

  return rows.map((row, index) => {
    const windowRows = rows.slice(Math.max(0, index - 19), index + 1);
    const valid = windowRows.map((item) => item.value).filter(Number.isFinite);
    const value = valid.length >= 5 ? valid.reduce((sum, item) => sum + item, 0) / valid.length : row.value;
    return { x: row.x, value };
  });
}

function buildResearchSummaryCard(report: ResearchReport, runId: string): FactorCard {
  const metrics = report.metrics || {};
  const alphaStats = report.alpha_stats || {};
  const passGates = Boolean(metrics.PassGates);
  return {
    run_id: report.run_id || runId,
    title: report.run_id || runId,
    status: passGates ? 'PASS' : 'RESEARCH',
    theme: passGates ? 'Submit-ready research report' : 'Research report summary',
    thesis: '该记录存在 research report，但未生成提交级 factor_card.json。这里展示当前可用的公式、评测指标、分布和诊断摘要。',
    definition: {
      formula: report.formula,
      inputs: [],
      update_frequency: '15-minute submission grid',
      postprocess: firstMetric(metrics, ['postprocess']) || '--',
    },
    metrics: {
      IC: Number(firstMetric(metrics, ['IC']) ?? 0),
      IR: Number(firstMetric(metrics, ['IR']) ?? 0),
      tvr: Number(firstMetric(metrics, ['tvr', 'TVR', 'Turnover']) ?? 0),
      Score: Number(firstMetric(metrics, ['Score', 'score']) ?? 0),
    },
    distribution: alphaStats,
    histogram: Array.isArray(alphaStats.histogram) ? alphaStats.histogram : [],
    prediction: {
      rolling_ic: rollingDailyIcSeries(report.daily_ic),
    },
    diagnostics: {
      daily_ic_count: report.n_daily_ic ?? Object.keys(report.daily_ic || {}).length,
      alpha_mean: alphaStats.mean ?? '--',
      alpha_std: alphaStats.std ?? '--',
      pct_positive: alphaStats.pct_positive ?? '--',
      pct_in_bounds: alphaStats.pct_in_bounds ?? '--',
    },
    gate_notes: passGates ? ['PassGates=true，满足当前提交门槛。'] : ['PassGates=false 或缺失，因此没有生成提交下载卡片。'],
    risk_notes: [],
    formula: report.formula,
    created_at: report.created_at,
  };
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

function compactDateLabel(value: any) {
  const text = String(value || '');
  const match = text.match(/\d{4}-\d{2}-\d{2}/);
  return match ? match[0] : text.slice(0, 16);
}

function niceUpperBound(value: number, minUpper: number, padding = 1.25) {
  const raw = Math.max(value * padding, minUpper);
  if (!Number.isFinite(raw) || raw <= 0) return minUpper;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalized = raw / magnitude;
  const step = normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return step * magnitude;
}

function isSubmitReady(factor: KbFactor) {
  return Boolean(factor.factor_card_path || factor.research_path);
}

function formatSignedNumber(value: number, digits = 2) {
  if (!Number.isFinite(value)) return '--';
  if (value > 0) return `+${value.toFixed(digits)}`;
  return value.toFixed(digits);
}

function metricFromPayload(payload: Record<string, any> | undefined, key: string) {
  if (!payload || typeof payload !== 'object') return 0;
  const raw = key === 'tvr' ? payload.tvr ?? payload.Turnover : payload[key];
  const value = Number(raw);
  return Number.isFinite(value) ? value : 0;
}

function periodMetricNumber(row: PeriodMetric | undefined, key: 'Score' | 'IC' | 'IR' | 'tvr', fallback = 0) {
  if (!row) return fallback;
  const raw = key === 'tvr' ? row.tvr ?? row.Turnover : row[key];
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

function factorPeriodRows(factor: KbFactor): PeriodMetric[] {
  if (Array.isArray(factor.period_metrics) && factor.period_metrics.length) {
    const order = ['discovery', 'oos_2024', 'full_2022_2024'];
    const baseRows = [...factor.period_metrics].sort((a, b) => order.indexOf(a.period) - order.indexOf(b.period));
    return baseRows;
  }
  const oosMetrics = factor.oss_2024?.metrics || {};
  const fullMetrics = factor.full_2022_2024?.metrics || {};
  const baseRows: PeriodMetric[] = [
    {
      period: 'discovery',
      label: '2022-2023',
      days: factor.eval_days,
      IC: factor.IC,
      IR: factor.IR,
      tvr: factor.tvr,
      Score: factor.Score,
      PassGates: factor.PassGates,
      used_for_discovery: true,
    },
    {
      period: 'oos_2024',
      label: '2024 OOS',
      days: factor.oss_2024?.days,
      IC: metricFromPayload(oosMetrics, 'IC'),
      IR: metricFromPayload(oosMetrics, 'IR'),
      tvr: metricFromPayload(oosMetrics, 'tvr'),
      Score: metricFromPayload(oosMetrics, 'Score'),
      PassGates: Boolean(oosMetrics.PassGates),
      used_for_discovery: false,
    },
    {
      period: 'full_2022_2024',
      label: '2022-2024',
      days: factor.full_2022_2024?.days,
      IC: metricFromPayload(fullMetrics, 'IC'),
      IR: metricFromPayload(fullMetrics, 'IR'),
      tvr: metricFromPayload(fullMetrics, 'tvr'),
      Score: metricFromPayload(fullMetrics, 'Score'),
      PassGates: Boolean(fullMetrics.PassGates),
      used_for_discovery: false,
    },
  ];
  return baseRows;
}

function periodLabel(row: PeriodMetric) {
  if (row.period === 'discovery') return '22-23';
  if (row.period === 'oos_2024') return 'OOS 2024';
  if (row.period === 'full_2022_2024') return 'Overall';
  return row.label || row.period;
}

function trendText(comparison?: OosComparison) {
  const trend = comparison?.trend;
  if (trend === 'improved') return 'OOS 增强';
  if (trend === 'stable') return 'OOS 稳定';
  if (trend === 'weaker') return 'OOS 走弱';
  return '待观察';
}

function deltaTone(value: number) {
  if (value > 0) return 'text-emerald-600';
  if (value < 0) return 'text-rose-600';
  return 'text-slate-500';
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

const SCREEN_THRESHOLDS = { IC: 0.02, ICIR: 1.0 };

function screenFailureDetails(factor: KbFactor): ScreenFailDetail[] {
  if (factor.screen_fail_details?.length) return factor.screen_fail_details;
  if (factor.status !== 'screened_out') return [];
  const details: ScreenFailDetail[] = [];
  if (Math.abs(factor.IC ?? 0) < SCREEN_THRESHOLDS.IC) {
    details.push({ key: 'IC', value: factor.IC, threshold: SCREEN_THRESHOLDS.IC, direction: 'abs >=', message: `|IC|=${formatNumber(Math.abs(factor.IC ?? 0), 4)}<${SCREEN_THRESHOLDS.IC}` });
  }
  if (Math.abs(factor.IR ?? 0) < SCREEN_THRESHOLDS.ICIR) {
    details.push({ key: 'ICIR', value: factor.IR, threshold: SCREEN_THRESHOLDS.ICIR, direction: 'abs >=', message: `|ICIR|=${formatNumber(Math.abs(factor.IR ?? 0), 3)}<${SCREEN_THRESHOLDS.ICIR}` });
  }
  const expectedDays = Number(factor.screening?.days || factor.eval_days || 0);
  const coveredDays = Number(factor.screening?.covered_days || factor.screening?.result_preview?.nd || factor.eval_days || 0);
  if (expectedDays > 0 && coveredDays > 0 && coveredDays < expectedDays) {
    details.push({ key: 'Days', value: coveredDays, threshold: expectedDays, direction: '>=', message: `Days=${coveredDays}/${expectedDays}` });
  }
  if (!details.length) details.push({ key: 'Score', value: factor.Score, threshold: 0, direction: '>', message: 'score=0' });
  return details;
}

function screenDetailLabel(detail: ScreenFailDetail) {
  const key = String(detail.key || '').toUpperCase();
  if (key === 'TVR' || key === 'TURNOVER') return 'Turnover 诊断';
  if (key === 'DAYS') return 'Days 未覆盖';
  if (key === 'COVERAGE') return '覆盖不足';
  if (key === 'IC') return 'IC 偏低';
  if (key === 'IR' || key === 'ICIR') return 'ICIR 偏低';
  if (key === 'SCORE') return 'Score 为 0';
  return String(detail.key || '未达标');
}

function screenDetailText(detail: ScreenFailDetail) {
  if (detail.message) return detail.message;
  const value = typeof detail.value === 'number' ? formatNumber(detail.value, detail.key === 'TVR' || detail.key === 'Days' ? 0 : 3) : detail.value;
  return `${detail.key}: ${value ?? '--'} ${detail.direction || ''} ${detail.threshold ?? ''}`.trim();
}

function factorFailureReason(factor: KbFactor) {
  if (factor.PassGates) return '通过';
  if (factor.status === 'invalid') return '公式无效';
  if (factor.status === 'compute_error') return '计算失败';
  if (factor.status === 'duplicate') return '重复结构';
  if (factor.status === 'screened_out') {
    return '快筛淘汰';
  }
  const failed = failedGateKeys(factor);
  if (failed.length) return failed.map((key) => key.toUpperCase()).join(' / ');
  if ((factor.Score ?? 0) <= 0 && factor.status === 'ok') {
    if (Math.abs(factor.IC ?? 0) <= 0.02) return 'IC 未过';
    if (Math.abs(factor.IR ?? 0) <= 1.0) return 'ICIR 未过';
    return 'Score 为 0';
  }
  return statusText(factor.status);
}

function gateTone(reason: string) {
  if (reason.includes('通过')) return 'bg-emerald-100 text-emerald-700';
  if (reason.includes('IC')) return 'bg-sky-100 text-sky-700';
  if (reason.includes('IR')) return 'bg-violet-100 text-violet-700';
  if (reason.includes('TVR') || reason.includes('Turnover')) return 'bg-orange-100 text-orange-700';
  if (reason.includes('TVR') && reason.includes('快筛')) return 'bg-orange-100 text-orange-700';
  if (reason.includes('IC') && reason.includes('快筛')) return 'bg-sky-100 text-sky-700';
  if (reason.includes('IR') && reason.includes('快筛')) return 'bg-violet-100 text-violet-700';
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
  if (factor.Score <= 0 || maxScore <= 0) return 'rgba(148, 163, 184, 0.04)';
  const alpha = Math.min(0.08 + (factor.Score / maxScore) * 0.2, 0.28);
  return `rgba(16, 185, 129, ${alpha})`;
}

const pillButtonClass = "inline-flex h-8 min-w-[4.75rem] items-center justify-center rounded-full px-3 text-xs font-medium";
const actionButtonClass = `${pillButtonClass} border border-border/60 text-foreground transition-colors hover:bg-white`;
const RECORDS_POLL_INTERVAL_MS = 15000;
const FACTOR_TABLE_BATCH_SIZE = 200;
const SUMMARY_QUERY = '/api/autoalpha/knowledge?compact_factors=1&include_factor_correlations=0&include_artifacts=0&include_generation_experiences=0';
const TABLE_QUERY = '/api/autoalpha/knowledge?include_factor_correlations=0&include_generation_experiences=0';
const CORRELATION_QUERY = '/api/autoalpha/factor-correlations';

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

const CorrelationTable = ({ rows }: { rows?: Array<Record<string, any>> }) => {
  const data = rows || [];
  if (!data.length) {
    return <div className="rounded-2xl bg-white/80 p-4 text-sm text-muted-foreground">暂无其他可提交因子可计算 realized alpha 相关性。</div>;
  }
  return (
    <div className="overflow-x-auto rounded-2xl border border-border/40 bg-white/85">
      <table className="min-w-[560px] table-fixed text-left text-xs">
        <colgroup>
          <col className="w-[310px]" />
          <col className="w-[80px]" />
          <col className="w-[80px]" />
          <col className="w-[90px]" />
        </colgroup>
        <thead className="bg-slate-50 text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
          <tr>
            <th className="px-3 py-2">Factor</th>
            <th className="px-3 py-2 text-right">Corr</th>
            <th className="px-3 py-2 text-right">Bars</th>
            <th className="px-3 py-2 text-right">Token</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row) => (
            <tr key={row.run_id} className="border-t border-border/30">
              <td className="truncate px-3 py-2 font-mono text-foreground" title={row.run_id}>{row.run_id}</td>
              <td className="px-3 py-2 text-right font-mono">{formatMetric(row.corr, 3)}</td>
              <td className="px-3 py-2 text-right font-mono">{row.n_bars ?? '--'}</td>
              <td className="px-3 py-2 text-right font-mono">{formatMetric(row.formula_token_overlap, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const RedundancySummary = ({ redundancy }: { redundancy?: Record<string, any> }) => {
  const metrics: Array<[string, any, number?]> = [
    ['Max Corr', redundancy?.max_alpha_corr, 3],
    ['Abs Corr', redundancy?.max_abs_alpha_corr, 3],
    ['Pairs', redundancy?.correlation_count, 0],
    ['Token', redundancy?.max_formula_token_overlap, 3],
  ];
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-2 xl:grid-cols-4">
        {metrics.map(([label, value, digits]) => (
          <div key={label} className="min-w-0 rounded-2xl bg-white/85 p-3">
            <div className="text-[11px] text-muted-foreground">{label}</div>
            <div className="mt-1 truncate font-mono text-sm font-semibold text-foreground" title={String(value ?? '')}>
              {formatMetric(value, digits)}
            </div>
          </div>
        ))}
      </div>
      <div className="space-y-2 rounded-2xl bg-white/85 p-3 text-xs leading-5">
        <div className="min-w-0">
          <div className="text-[11px] text-muted-foreground">Nearest Token</div>
          <div className="mt-1 truncate font-mono font-semibold text-foreground" title={redundancy?.nearest_factor || '--'}>
            {redundancy?.nearest_factor || '--'}
          </div>
        </div>
        <div className="min-w-0">
          <div className="text-[11px] text-muted-foreground">Family</div>
          <div className="mt-1 break-words font-mono font-semibold text-foreground">
            {redundancy?.family || '--'}
          </div>
        </div>
      </div>
    </div>
  );
};

const CorrelationHeatmap = ({
  labels,
  runIds,
  matrix,
  factorOrdinalByRunId,
  onSelectRunId,
}: {
  labels: string[];
  runIds?: string[];
  matrix: (number | null)[][];
  factorOrdinalByRunId: Map<string, number>;
  onSelectRunId: (runId: string) => void;
}) => {
  if (!labels.length || !matrix.length) {
    return <div className="flex h-full items-center justify-center text-xs text-muted-foreground">暂无相关性数据</div>;
  }
  const rawIds = runIds?.length ? runIds : labels;
  // Re-sort rows/columns by chronological generation order, independent of backend cache order.
  const sortedPerm = rawIds
    .map((id, i) => ({ i, ord: factorOrdinalByRunId.get(id) ?? i + 1 }))
    .sort((a, b) => a.ord - b.ord)
    .map((x) => x.i);
  const ids = sortedPerm.map((i) => rawIds[i]);
  const sortedMatrix = sortedPerm.map((ri) => sortedPerm.map((ci) => matrix[ri]?.[ci] ?? null));
  const cellColor = (v: number | null) => {
    if (v === null || v === undefined) return '#f8fafc';
    const abs = Math.abs(v);
    if (abs >= 0.7) return v > 0 ? '#dc2626' : '#2563eb';
    if (abs >= 0.4) return v > 0 ? '#f97316' : '#7c3aed';
    if (abs >= 0.2) return v > 0 ? '#fbbf24' : '#06b6d4';
    return '#f1f5f9';
  };
  const textColor = (v: number | null) => {
    if (v === null || v === undefined) return '#94a3b8';
    return Math.abs(v) >= 0.4 ? '#fff' : '#334155';
  };
  const orderLabel = (runId: string, fallbackIndex: number) => String(factorOrdinalByRunId.get(runId) ?? fallbackIndex + 1);
  const factorLabel = (runId: string, fallbackIndex: number) => `#${orderLabel(runId, fallbackIndex)}`;
  return (
    <div className="overflow-auto">
      <table className="text-[10px] border-collapse w-full">
        <thead>
          <tr>
            <th className="p-1 text-left text-muted-foreground w-8"></th>
            {ids.map((runId, index) => (
              <th
                key={runId}
                className="p-1 text-center text-muted-foreground font-normal"
                title={`第 ${orderLabel(runId, index)} 个生成因子 · ${runId} · 点击打开因子卡片`}
              >
                <button
                  type="button"
                  onClick={() => onSelectRunId(runId)}
                  className="rounded px-1 font-mono text-[10px] text-sky-700 underline decoration-sky-300 underline-offset-2 hover:text-sky-900"
                >
                  {factorLabel(runId, index)}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedMatrix.map((row, ri) => (
            <tr key={ids[ri] ?? ri}>
              <td
                className="p-1 text-muted-foreground whitespace-nowrap"
                title={`第 ${orderLabel(ids[ri], ri)} 个生成因子 · ${ids[ri]} · 点击打开因子卡片`}
              >
                <button
                  type="button"
                  onClick={() => onSelectRunId(ids[ri])}
                  className="rounded px-1 font-mono text-[10px] text-sky-700 underline decoration-sky-300 underline-offset-2 hover:text-sky-900"
                >
                  {factorLabel(ids[ri], ri)}
                </button>
              </td>
              {row.map((val, ci) => (
                <td
                  key={ci}
                  title={`${factorLabel(ids[ri], ri)} (${ids[ri]}) × ${factorLabel(ids[ci], ci)} (${ids[ci]}): ${val !== null ? val.toFixed(3) : 'N/A'}`}
                  style={{ backgroundColor: cellColor(val), color: textColor(val) }}
                  className="p-0.5 text-center font-mono font-semibold transition-all"
                >
                  {val !== null && val !== undefined ? val.toFixed(2) : '—'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

function buildFactorCorrelationTrend(payload: FactorCorrelationPayload | undefined | null, factors: KbFactor[]) {
  if (!payload?.matrix?.length && !payload?.trend_rows?.length) return [];

  const factorMetaById = new Map(
    [...factors]
      .sort(
        (a, b) =>
          String(a.created_at || '').localeCompare(String(b.created_at || '')) ||
          String(a.run_id || '').localeCompare(String(b.run_id || ''))
      )
      .map((factor, index) => [
        factor.run_id,
        {
          generation: Number(factor.generation || 0),
          testedIndex: index + 1,
        },
      ])
  );

  if (payload.trend_rows?.length) {
    return payload.trend_rows
      .filter((row) => row && row.run_id)
      .sort((a, b) => Number(a.index || 0) - Number(b.index || 0))
      .map((row, orderIndex) => ({
        index: Number(row.index || orderIndex + 1),
        label: row.label || `#${Number(row.index || orderIndex + 1)}`,
        run_id: row.run_id,
        generation: Number(factorMetaById.get(row.run_id)?.generation || row.generation || 0),
        tested_index: Number(factorMetaById.get(row.run_id)?.testedIndex || row.tested_index || 0),
        avg_corr: Number(Number(row.avg_corr || 0).toFixed(4)),
        max_corr: Number(Number(row.max_corr || 0).toFixed(4)),
        min_corr: Number(Number(row.min_corr || 0).toFixed(4)),
      }));
  }

  const runIds = payload.run_ids?.length ? payload.run_ids : payload.labels;
  const sourceIds = runIds;

  return sourceIds
    .map((runId) => runIds.indexOf(runId))
    .filter((index) => index >= 0)
    .map((matrixIndex, orderIndex) => {
      const values = (payload.matrix[matrixIndex] || [])
        .map((value, index) => (index === matrixIndex || value == null ? null : Number(value)))
        .filter((value): value is number => value !== null && Number.isFinite(value));
      if (!values.length) return null;
      const avg = values.reduce((sum, value) => sum + value, 0) / values.length;
      return {
        index: orderIndex + 1,
        label: `#${orderIndex + 1}`,
        run_id: runIds[matrixIndex],
        generation: Number(factorMetaById.get(runIds[matrixIndex])?.generation || 0),
        tested_index: Number(factorMetaById.get(runIds[matrixIndex])?.testedIndex || 0),
        avg_corr: Number(avg.toFixed(4)),
        max_corr: Number(Math.max(...values).toFixed(4)),
        min_corr: Number(Math.min(...values).toFixed(4)),
      };
    })
    .filter((row): row is NonNullable<typeof row> => Boolean(row));
}

function hasHeatmapMatrix(payload: FactorCorrelationPayload | undefined | null) {
  return Boolean(payload?.labels?.length && payload?.matrix?.length);
}

const FactorCorrelationTrendChart = ({
  factorCorrelations,
  factors,
}: {
  factorCorrelations?: FactorCorrelationPayload;
  factors: KbFactor[];
}) => {
  const rows = useMemo(
    () => buildFactorCorrelationTrend(factorCorrelations, factors),
    [factorCorrelations, factors]
  );

  if (!rows.length) {
    return <div className="flex h-full items-center justify-center rounded-2xl bg-slate-50 text-xs text-muted-foreground">暂无有效因子相关性时序</div>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={rows}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
        <XAxis dataKey="index" tick={{ fontSize: 11 }} tickFormatter={(value) => `#${value}`} minTickGap={18} />
        <YAxis domain={['dataMin - 0.05', 1]} tick={{ fontSize: 11 }} />
        <Tooltip
          formatter={(value: any) => formatMetric(value, 4)}
          labelFormatter={(_, payload) => {
            const row = payload?.[0]?.payload as { run_id?: string; generation?: number; tested_index?: number } | undefined;
            if (!row) return '';
            const genLabel = row.generation && row.tested_index ? `Gen ${row.generation}/${row.tested_index}` : '';
            return [row.run_id, genLabel].filter(Boolean).join(' · ');
          }}
        />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line type="monotone" dataKey="max_corr" name="最大相关性" stroke="#ef4444" strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="avg_corr" name="平均相关性" stroke="#2563eb" strokeWidth={2.5} dot={{ r: 2 }} />
        <Line type="monotone" dataKey="min_corr" name="最小相关性" stroke="#10b981" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
};

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
  if (source === 'systematic') return 'Systematic';
  if (source === 'manual') return 'Manual';
  if (source === 'paper') return 'Paper';
  if (source === 'llm') return 'LLM';
  return source;
};

const ALL_SOURCES = ['systematic', 'manual', 'paper', 'llm'] as const;

function buildMarketStateChartData(card?: FactorCard) {
  const monthly = card?.monthly_ic?.monthly || [];
  if (monthly.length) {
    const rows = monthly.map((row: Record<string, any>) => ({
      x: compactDateLabel(row.x),
      ic: row.value,
      ...(row.market_ret != null ? { market_ret: row.market_ret } : {}),
      ...(row.market_vol != null ? { market_vol: row.market_vol } : {}),
      ...(row.market_abs_move != null ? { market_abs_move: row.market_abs_move } : {}),
    }));
    if (rows.some((row: Record<string, any>) => row.market_ret != null || row.market_vol != null || row.market_abs_move != null)) {
      return rows;
    }
  }
  return (card?.temporal?.market_state || [])
    .map((row: Record<string, any>) => ({
      x: compactDateLabel(row.x),
      ...(row.ic != null ? { ic: row.ic } : {}),
      ...(row.price != null ? { market_ret: row.price } : {}),
      ...(row.volatility != null ? { market_vol: row.volatility } : {}),
      ...(row.liquidity != null ? { market_abs_move: row.liquidity } : {}),
    }))
    .filter((row: Record<string, any>) => row.ic != null || row.market_ret != null || row.market_vol != null);
}

function buildPeriodEvolution(factors: KbFactor[]) {
  const ordered = [...factors].sort((a, b) => String(a.created_at || '').localeCompare(String(b.created_at || '')));
  let bestTrainScore = 0;
  let bestOosScore = 0;
  let bestTrainIc = 0;
  let bestOosIc = 0;
  return ordered.map((factor, index) => {
    const rows = factorPeriodRows(factor);
    const train = rows.find((row) => row.period === 'discovery');
    const oos = rows.find((row) => row.period === 'oos_2024');
    const passes = factor.PassGates === true || String(factor.PassGates).toLowerCase() === 'true';
    if (passes) {
      bestTrainScore = Math.max(bestTrainScore, periodMetricNumber(train, 'Score', Number(factor.Score || 0)));
      bestOosScore = Math.max(bestOosScore, periodMetricNumber(oos, 'Score', metricFromPayload(factor.oss_2024?.metrics, 'Score')));
      bestTrainIc = Math.max(bestTrainIc, periodMetricNumber(train, 'IC', Number(factor.IC || 0)));
      bestOosIc = Math.max(bestOosIc, periodMetricNumber(oos, 'IC', metricFromPayload(factor.oss_2024?.metrics, 'IC')));
    }
    return {
      index: index + 1,
      label: `#${index + 1}`,
      train_score: Number(bestTrainScore.toFixed(3)),
      oos_score: Number(bestOosScore.toFixed(3)),
      train_ic: Number(bestTrainIc.toFixed(4)),
      oos_ic: Number(bestOosIc.toFixed(4)),
    };
  });
}

const PeriodEvolutionCharts = ({ factors }: { factors: KbFactor[] }) => {
  const rows = useMemo(() => buildPeriodEvolution(factors), [factors]);
  if (!rows.length) {
    return <div className="flex h-full items-center justify-center rounded-2xl bg-slate-50 text-xs text-muted-foreground">暂无 IS/OOS 趋势数据</div>;
  }
  const scoreMax = niceUpperBound(Math.max(...rows.map((row) => Math.max(row.train_score, row.oos_score)), 0), 1);
  const icMax = niceUpperBound(Math.max(...rows.map((row) => Math.max(row.train_ic, row.oos_ic)), 0), 1);
  return (
    <div className="grid h-full gap-3 md:grid-cols-2">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
          <XAxis dataKey="label" tick={{ fontSize: 11 }} minTickGap={18} />
          <YAxis domain={[0, scoreMax]} tick={{ fontSize: 11 }} />
          <Tooltip formatter={(value: any) => formatMetric(value, 2)} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line type="monotone" dataKey="train_score" name="IS Score" stroke="#0f766e" strokeWidth={2.5} dot={false} />
          <Line type="monotone" dataKey="oos_score" name="OOS Score" stroke="#dc2626" strokeWidth={2.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
          <XAxis dataKey="label" tick={{ fontSize: 11 }} minTickGap={18} />
          <YAxis domain={[0, icMax]} tick={{ fontSize: 11 }} />
          <Tooltip formatter={(value: any) => formatMetric(value, 3)} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line type="monotone" dataKey="train_ic" name="IS IC" stroke="#2563eb" strokeWidth={2.5} dot={false} />
          <Line type="monotone" dataKey="oos_ic" name="OOS IC" stroke="#f97316" strokeWidth={2.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

const InspirationStatsCharts = ({
  stats,
  factors,
  factorCorrelations,
  correlationLoading,
  correlationError,
  onSelectRunId,
}: {
  stats?: KnowledgePayload['inspiration_stats'];
  factors: KbFactor[];
  factorCorrelations?: FactorCorrelationPayload;
  correlationLoading: boolean;
  correlationError: string;
  onSelectRunId: (runId: string) => void;
}) => {
  const timeline = stats?.timeline || [];
  const sourceColors: Record<string, string> = {
    systematic: '#0f766e',
    manual: '#2563eb',
    paper: '#059669',
    llm: '#7c3aed',
  };
  const heatmapData = hasHeatmapMatrix(factorCorrelations) ? (factorCorrelations ?? null) : null;

  const factorOrdinalByRunId = useMemo(
    () =>
      new Map(
        [...factors]
          .sort(
            (a, b) =>
              String(a.created_at || '').localeCompare(String(b.created_at || '')) ||
              String(a.run_id || '').localeCompare(String(b.run_id || ''))
          )
          .map((factor, index) => [factor.run_id, index + 1])
      ),
    [factors]
  );

  const bySource = useMemo(() => {
    const map = new Map((stats?.by_source || []).map((s) => [s.source, s]));
    return ALL_SOURCES.map((source) => map.get(source) ?? {
      source, prompt_count: 0, tested_count: 0, passing_count: 0,
      pass_rate: 0, valid_per_prompt: 0, valid_share: 0,
    });
  }, [stats?.by_source]);

  const sourceMixAdvice = useMemo(() => {
    const rows = new Map(bySource.map((row) => [row.source, row]));
    const paperTested = Number(rows.get('paper')?.tested_count || 0);
    const llmTested = Number(rows.get('llm')?.tested_count || 0);
    const systematicPassing = Number(rows.get('systematic')?.passing_count || 0);
    const notes: string[] = [];
    if (systematicPassing > 0) notes.push('Systematic 是 v3 冷启动来源，已从 Manual 中拆分。');
    if (paperTested < 3) notes.push('Paper 来源偏少：建议提高 fetch cycle 的 paper 配额，并优先让 LLM 从论文机制改写公式。');
    if (llmTested < 3) notes.push('LLM 来源偏少：建议保留探索位给 LLM brainstorm，不从已通过系统化家族继续局部变体。');
    return notes;
  }, [bySource]);

  if (!stats) {
    return <div className="rounded-3xl bg-white/80 p-5 text-sm text-muted-foreground">暂无灵感源统计。抓取或同步灵感后，这里会显示来源转化率。</div>;
  }

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <div className="rounded-3xl border border-border/50 bg-white/90 p-4">
        <div className="mb-3">
          <div className="text-sm font-medium text-foreground">灵感源转化</div>
          <div className="mt-1 text-xs text-muted-foreground">左轴为真实因子记录数 / passing 因子数，右轴为通过率。</div>
          {stats.source_mix_note ? <div className="mt-1 text-[11px] text-muted-foreground">{stats.source_mix_note}</div> : null}
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
              <Bar yAxisId="left" dataKey="tested_count" name="因子记录数" fill="#94a3b8" radius={[4, 4, 0, 0]} />
              <Bar yAxisId="left" dataKey="passing_count" name="Passing 因子" fill="#10b981" radius={[4, 4, 0, 0]} />
              <Line yAxisId="right" type="monotone" dataKey="pass_rate" name="通过率 %" stroke="#ef4444" strokeWidth={2} dot />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        {sourceMixAdvice.length ? (
          <div className="mt-3 space-y-1 rounded-2xl bg-slate-50 p-3 text-xs leading-5 text-slate-700">
            {sourceMixAdvice.map((note) => <div key={note}>{note}</div>)}
          </div>
        ) : null}
      </div>

      <div className="rounded-3xl border border-border/50 bg-white/90 p-4">
        <div className="mb-3">
          <div className="text-sm font-medium text-foreground">最佳 Score / IC 趋势</div>
          <div className="mt-1 text-xs text-muted-foreground">左图为累计最佳 Score，右图为累计最佳 IC；每张图都分开显示 22-23 IS 与 2024 OOS。</div>
        </div>
        <div className="h-[280px]">
          <PeriodEvolutionCharts factors={factors} />
        </div>
      </div>

      <div className="rounded-3xl border border-border/50 bg-white/90 p-4">
        <div className="mb-3">
          <div className="text-sm font-medium text-foreground">通过率随 Record 变化</div>
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
              {ALL_SOURCES.map((source) => (
                <Line key={source} type="monotone" dataKey={`${source}_pass_rate`} name={`${sourceLabel(source)} 通过率`} stroke={sourceColors[source]} strokeWidth={2} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="rounded-3xl border border-border/50 bg-white/90 p-4">
        <div className="mb-3">
          <div className="text-sm font-medium text-foreground">因子相关性热图</div>
          <div className="mt-1 text-xs text-muted-foreground">所有通过因子两两 alpha 相关系数；横纵坐标显示生成序号，点击序号可直接打开对应因子卡片。颜色越深相关性越高，红为正相关，蓝为负相关。</div>
          {factorCorrelations?.meta ? (
            <div className="mt-1 text-[11px] text-muted-foreground">
              热图缓存 {factorCorrelations.meta.actual_heatmap_count ?? 0}/{factorCorrelations.meta.expected_heatmap_count ?? 0}
              {factorCorrelations.meta.refreshing ? '，后台刷新中…' : ''}
            </div>
          ) : null}
        </div>
        <div className="h-[240px] overflow-auto">
          {correlationLoading ? (
            <div className="flex h-full items-center justify-center text-xs text-muted-foreground">相关性数据加载中…</div>
          ) : correlationError ? (
            <div className="flex h-full items-center justify-center rounded-2xl bg-red-50 px-4 text-xs text-red-600">{correlationError}</div>
          ) : heatmapData ? (
            <CorrelationHeatmap
              labels={heatmapData.labels}
              runIds={heatmapData.run_ids}
              matrix={heatmapData.matrix}
              factorOrdinalByRunId={factorOrdinalByRunId}
              onSelectRunId={onSelectRunId}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-xs text-muted-foreground">加载中…</div>
          )}
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
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 12000);
    setReport(null);
    setError('');

    fetchJson<{ report: ResearchReport }>(`/api/autoalpha/research/${runId}?full=1`, { signal: controller.signal })
      .then((data) => {
        if (mounted) setReport(data.report);
      })
      .catch((err: Error) => {
        if (!mounted) return;
        const message =
          err.name === 'AbortError'
            ? 'Factor Card 请求超时。通常是 8080 上还在跑旧后端，或后端接口卡住了。请重启 backend 后再试。'
            : err.message;
        setError(message);
      });
    return () => {
      mounted = false;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [runId]);

  const nativeCard = report?.factor_card;
  const card = report ? nativeCard || buildResearchSummaryCard(report, runId) : null;
  const cardMetrics = card?.metrics || {};
  const diagnostics = card?.diagnostics || {};
  const canDownload = Boolean(nativeCard?.status === 'PASS' || report?.metrics?.PassGates);
  const cardIsPass = card?.status === 'PASS';
  const shellClass = cardIsPass
    ? 'border-emerald-200 bg-emerald-50/60'
    : 'border-slate-200 bg-slate-50';
  const statusClass = cardIsPass ? 'bg-emerald-600 text-white' : 'bg-slate-200 text-slate-700';
  const eyebrowClass = cardIsPass ? 'text-emerald-700' : 'text-slate-500';
  const themeClass = cardIsPass ? 'text-emerald-800' : 'text-slate-600';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4" onClick={onClose}>
      <div className="max-h-[88vh] w-full max-w-4xl overflow-y-auto rounded-[28px] border border-border/50 bg-white p-6 shadow-2xl" onClick={(event) => event.stopPropagation()}>
        <div className="mb-5 flex items-center justify-between gap-3">
          <div>
            <div className="text-[11px] uppercase tracking-[0.22em] text-muted-foreground">Factor Card</div>
            <div className="mt-2 text-xl font-semibold text-foreground">{runId}</div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {canDownload ? (
              <a href={`/api/autoalpha/factors/${encodeURIComponent(runId)}/download`} className={`${actionButtonClass} gap-1.5`} title="下载提交 parquet 与 JSON">
                <Download className="h-3.5 w-3.5" />
                下载
              </a>
            ) : null}
            <button onClick={onClose} className={actionButtonClass}>
              关闭
            </button>
          </div>
        </div>
        {error ? <div className="rounded-2xl bg-red-50 p-4 text-sm text-red-600">{error}</div> : null}
        {!report && !error ? <div className="text-sm text-muted-foreground">Factor Card 加载中...</div> : null}
        {report ? (
          <div className="space-y-4">
            {card ? (
              <div className={`space-y-4 rounded-3xl border p-4 ${shellClass}`}>
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className={`text-[11px] uppercase tracking-[0.22em] ${eyebrowClass}`}>
                      {nativeCard ? 'Factor Card' : 'Research Summary'}
                    </div>
                    <div className="mt-2 break-all text-lg font-semibold text-slate-950">{card.title || card.run_id}</div>
                    {card.theme ? <div className={`mt-1 text-sm ${themeClass}`}>{card.theme}</div> : null}
                  </div>
                  <div className={`rounded-full px-3 py-1 text-xs font-semibold ${statusClass}`}>
                    {card.status}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  {[
                    ['IC', firstMetric(cardMetrics, ['IC'])],
                    ['IR', firstMetric(cardMetrics, ['IR'])],
                    ['TVR', firstMetric(cardMetrics, ['tvr', 'TVR', 'Turnover'])],
                    ['Score', firstMetric(cardMetrics, ['Score', 'score'])],
                  ].map(([label, value]) => (
                    <div key={String(label)} className="rounded-2xl bg-white/90 p-3">
                      <div className="text-[11px] text-muted-foreground">{label}</div>
                      <div className="mt-2 font-mono text-lg font-semibold text-foreground">{formatMetric(value)}</div>
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
                    {card.thesis ? <div>{card.thesis}</div> : null}
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
                <CardSection title="3. 时序样貌">
                  <SmallChart title="20 日滚动 IC" description="核心预测力是否持续；穿 0 代表近期方向可能失效。" data={card.prediction?.rolling_ic} color="#0f766e" height={200} />
                </CardSection>
                <CardSection title="4. 基本分析">
                  <div className="space-y-3">
                    <MultiLineChart
                      title="市场状态 IC 对比"
                      description="月度 IC（紫）与可用市场状态序列的对比；用于观察因子是否集中在某些市场环境中生效。"
                      data={buildMarketStateChartData(card)}
                      lines={[
                        { key: 'ic', name: '月度 IC', color: '#9333ea' },
                        { key: 'market_ret', name: '市场收益', color: '#10b981' },
                        { key: 'market_vol', name: '市场波动', color: '#f97316' },
                        { key: 'market_abs_move', name: '市场振幅', color: '#0ea5e9' },
                      ]}
                      height={220}
                    />
                    {(card.regime || []).length > 0 && (
                      <div className="grid gap-2 grid-cols-3 md:grid-cols-6">
                        {(card.regime || []).map((row: Record<string, any>) => (
                          <div key={String(row.regime)} className="rounded-2xl bg-white/85 p-2 text-center">
                            <div className="text-[10px] text-muted-foreground">{String(row.regime)}</div>
                            <div className="mt-1 font-mono text-xs font-semibold">IC {formatMetric(row.ic)}</div>
                            <div className="text-[10px] text-muted-foreground">{row.days}d</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </CardSection>
                <CardSection title="5. 相关性与冗余">
                  <div className="grid min-w-0 gap-3 xl:grid-cols-[0.9fr_1.5fr]">
                    <RedundancySummary redundancy={card.redundancy} />
                    <CorrelationTable rows={card.redundancy?.top_alpha_correlations} />
                  </div>
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

export const AutoAlphaRecordsPage: React.FC = () => {
  const [summaryKnowledge, setSummaryKnowledge] = useState<KnowledgePayload | null>(null);
  const [tableKnowledge, setTableKnowledge] = useState<KnowledgePayload | null>(null);
  const [factorCorrelations, setFactorCorrelations] = useState<FactorCorrelationPayload | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(true);
  const [tableLoading, setTableLoading] = useState(true);
  const [correlationLoading, setCorrelationLoading] = useState(true);
  const [summaryError, setSummaryError] = useState('');
  const [tableError, setTableError] = useState('');
  const [correlationError, setCorrelationError] = useState('');
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [generationNote, setGenerationNote] = useState<GenerationExperience | null>(null);
  const [generationNoteLoading, setGenerationNoteLoading] = useState<number | null>(null);
  const [generationNoteRegenerating, setGenerationNoteRegenerating] = useState(false);
  const [visibleFactorCount, setVisibleFactorCount] = useState(FACTOR_TABLE_BATCH_SIZE);
  const lastSummaryVersionRef = useRef('');
  const lastTableVersionRef = useRef('');

  useEffect(() => {
    let mounted = true;
    let inflight = false;
    const load = async () => {
      if (inflight || document.hidden) return;
      inflight = true;
      if (mounted) setSummaryLoading((current) => current && !summaryKnowledge);
      try {
        const knowledgeData = await fetchJson<KnowledgePayload>(SUMMARY_QUERY);
        if (mounted) {
          const version = [
            knowledgeData.updated_at,
            knowledgeData.total_tested,
            knowledgeData.total_passing,
            knowledgeData.best_score,
          ].join('|');
          if (lastSummaryVersionRef.current !== version) {
            lastSummaryVersionRef.current = version;
            setSummaryKnowledge(knowledgeData);
          }
          setSummaryError('');
          setSummaryLoading(false);
        }
      } catch (error: any) {
        if (mounted && !summaryKnowledge) {
          setSummaryError(error?.message || 'Summary 加载失败');
          setSummaryLoading(false);
        }
      } finally {
        inflight = false;
      }
    };
    load();
    const handleVisibilityChange = () => {
      if (!document.hidden) load();
    };
    const timer = window.setInterval(load, RECORDS_POLL_INTERVAL_MS);
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      mounted = false;
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [summaryKnowledge]);

  useEffect(() => {
    let mounted = true;
    let inflight = false;
    const load = async () => {
      if (inflight || document.hidden) return;
      inflight = true;
      if (mounted) setTableLoading((current) => current && !tableKnowledge);
      try {
        const knowledgeData = await fetchJson<KnowledgePayload>(TABLE_QUERY);
        if (mounted) {
          const version = [
            knowledgeData.updated_at,
            knowledgeData.total_tested,
            knowledgeData.total_passing,
            knowledgeData.best_score,
            knowledgeData.factors.length,
            knowledgeData.artifacts.output_files.length,
            knowledgeData.artifacts.research_reports.length,
          ].join('|');
          if (lastTableVersionRef.current !== version) {
            lastTableVersionRef.current = version;
            setTableKnowledge(knowledgeData);
          }
          setTableError('');
          setTableLoading(false);
        }
      } catch (error: any) {
        if (mounted && !tableKnowledge) {
          setTableError(error?.message || 'Record 表格加载失败');
          setTableLoading(false);
        }
      } finally {
        inflight = false;
      }
    };
    load();
    const handleVisibilityChange = () => {
      if (!document.hidden) load();
    };
    const timer = window.setInterval(load, RECORDS_POLL_INTERVAL_MS);
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      mounted = false;
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [tableKnowledge]);

  useEffect(() => {
    let mounted = true;
    let inflight = false;
    const load = async () => {
      if (inflight || document.hidden) return;
      inflight = true;
      if (mounted) setCorrelationLoading((current) => current && !factorCorrelations);
      try {
        const data = await fetchJson<FactorCorrelationPayload>(CORRELATION_QUERY);
        if (mounted) {
          setFactorCorrelations(data);
          setCorrelationError('');
          setCorrelationLoading(false);
        }
      } catch (error: any) {
        if (mounted && !factorCorrelations) {
          setCorrelationError(error?.message || '相关性缓存加载失败');
          setCorrelationLoading(false);
        }
      } finally {
        inflight = false;
      }
    };
    load();
    const handleVisibilityChange = () => {
      if (!document.hidden) load();
    };
    const timer = window.setInterval(load, RECORDS_POLL_INTERVAL_MS);
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      mounted = false;
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  const summaryFactors = summaryKnowledge?.factors ?? [];
  const factors = tableKnowledge?.factors ?? [];
  const deferredFactors = useDeferredValue(factors);
  const factorOrdinalByRunId = useMemo(
    () =>
      new Map(
        [...deferredFactors]
          .sort(
            (a, b) =>
              String(a.created_at || '').localeCompare(String(b.created_at || '')) ||
              String(a.run_id || '').localeCompare(String(b.run_id || ''))
          )
          .map((factor, index) => [factor.run_id, index + 1])
      ),
    [deferredFactors]
  );
  const outputFiles = tableKnowledge?.artifacts.output_files ?? [];
  const researchReports = tableKnowledge?.artifacts.research_reports ?? [];
  const visibleFactors = useMemo(
    () => deferredFactors.slice(0, visibleFactorCount),
    [deferredFactors, visibleFactorCount]
  );
  const oosComparisonFactors = useMemo(
    () =>
      deferredFactors
        .filter((factor) => factor.PassGates && factor.oss_2024?.metrics)
        .slice(0, 12),
    [deferredFactors]
  );
  const maxScore = Math.max(...visibleFactors.map((factor) => factor.Score), 1);
  const lineageLanes = useMemo(
    () =>
      Array.from(
        deferredFactors.reduce((acc, factor) => {
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
    [deferredFactors]
  );

  useEffect(() => {
    setVisibleFactorCount((current) => {
      if (!deferredFactors.length) return FACTOR_TABLE_BATCH_SIZE;
      const clamped = Math.min(current, deferredFactors.length);
      return Math.max(Math.min(FACTOR_TABLE_BATCH_SIZE, deferredFactors.length), clamped);
    });
  }, [deferredFactors.length]);

  const handleOpenGenerationNote = async (generation: number) => {
    try {
      setGenerationNoteLoading(generation);
      const note = await fetchJson<GenerationExperience>(`/api/autoalpha/generation-experience/${generation}`);
      setGenerationNote(note);
    } catch {
      // No experience yet — open modal in empty state so user can trigger generation manually
      setGenerationNote({ generation, created_at: '', path: '', relative_path: '', summary: '' });
    } finally {
      setGenerationNoteLoading(null);
    }
  };

  const handleRegenerateGenerationNote = async () => {
    if (!generationNote) return;
    try {
      setGenerationNoteRegenerating(true);
      const note = await fetchJson<GenerationExperience>(
        `/api/autoalpha/generation-experience/${generationNote.generation}`,
        { method: 'POST' }
      );
      setGenerationNote(note);
    } catch (error: any) {
      alert(error.message || '重新生成失败');
    } finally {
      setGenerationNoteRegenerating(false);
    }
  };


  return (
    <div className="space-y-6 pb-10">
      <Panel title="AutoAlpha Record" subtitle="Generation 演进、产出文件、因子卡片和知识库因子表">
        {summaryError && !summaryKnowledge ? (
          <div className="rounded-3xl bg-red-50 p-5 text-sm text-red-600">{summaryError}</div>
        ) : (
          <div className="grid gap-4 md:grid-cols-5">
            <div className="rounded-3xl bg-white/80 p-4">
              <div className="text-xs text-muted-foreground">已测试</div>
              <div className="mt-2 text-2xl font-semibold">{summaryKnowledge?.total_tested ?? 0}</div>
            </div>
            <div className="rounded-3xl bg-emerald-50 p-4">
              <div className="text-xs text-muted-foreground">通过 Gate</div>
              <div className="mt-2 text-2xl font-semibold">{summaryKnowledge?.total_passing ?? 0}</div>
            </div>
            <div className="rounded-3xl bg-amber-50 p-4">
              <div className="text-xs text-muted-foreground">通过率</div>
              <div className="mt-2 text-2xl font-semibold">{formatNumber(summaryKnowledge?.pass_rate ?? 0, 1)}%</div>
            </div>
            <div className="rounded-3xl bg-sky-50 p-4">
              <div className="text-xs text-muted-foreground">最佳 Score</div>
              <div className="mt-2 text-2xl font-semibold">{formatNumber(summaryKnowledge?.best_score ?? 0, 2)}</div>
            </div>
            <div className="rounded-3xl bg-violet-50 p-4">
              <div className="text-xs text-muted-foreground">最近更新</div>
              <div className="mt-2 text-lg font-semibold">{formatDateTime(summaryKnowledge?.updated_at)}</div>
            </div>
          </div>
        )}
        {summaryLoading && !summaryKnowledge ? (
          <div className="mt-3 text-xs text-muted-foreground">Summary 加载中…</div>
        ) : null}
      </Panel>

      <Panel title="灵感源转化分析" subtitle="按 Manual、Paper 和 LLM 追踪真实因子记录数、passing 因子数、通过率与有效因子贡献。">
        <InspirationStatsCharts
          stats={summaryKnowledge?.inspiration_stats}
          factors={deferredFactors.length ? deferredFactors : summaryFactors}
          factorCorrelations={factorCorrelations ?? undefined}
          correlationLoading={correlationLoading}
          correlationError={correlationError}
          onSelectRunId={setSelectedRunId}
        />
      </Panel>

      <Panel title="Evolutionary Generation">
        <div className="max-h-[620px] overflow-auto rounded-3xl border border-border/40 bg-white/70 p-3">
          <div className="flex min-w-max items-start gap-4 pb-2">
            {tableLoading && !tableKnowledge ? (
              <div className="text-sm text-muted-foreground">Record 表格加载中…</div>
            ) : lineageLanes.length === 0 ? (
              <div className="text-sm text-muted-foreground">还没有代际数据。</div>
            ) : (
              lineageLanes.map((lane, index) => (
                <div key={lane.generation} className="flex items-start gap-4">
                  <div className="w-[280px]">
                    <div className="mb-3 flex items-center justify-between gap-2 text-sm font-medium text-foreground">
                      <div className="flex min-w-0 items-center gap-2">
                        <GitBranch className="h-4 w-4 shrink-0 text-emerald-500" />
                        <span className="truncate">Generation {lane.generation}</span>
                      </div>
                      <button
                        onClick={() => handleOpenGenerationNote(lane.generation)}
                        disabled={generationNoteLoading === lane.generation}
                        className="inline-flex h-8 shrink-0 items-center gap-1 rounded-full border border-sky-200 bg-sky-50 px-2 text-[11px] font-medium text-sky-700 transition-colors hover:bg-sky-100 disabled:opacity-50"
                        title="查看 Generation 经验总结"
                      >
                        <FileText className="h-3.5 w-3.5" />
                        {generationNoteLoading === lane.generation ? '加载中' : '经验'}
                      </button>
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

      <Panel title="因子与日志">
        {tableError && !tableKnowledge ? (
          <div className="mb-4 rounded-3xl bg-red-50 p-5 text-sm text-red-600">{tableError}</div>
        ) : null}
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-3xl border border-border/50 bg-white/90 p-4">
            <div className="mb-3">
              <div className="text-sm font-medium text-foreground">有效因子相关性时序</div>
              <div className="mt-1 text-xs text-muted-foreground">横轴为通过 Gate 的出现顺序；每个点表示该因子在出现当时相对之前已通过因子的相关性统计，趋势序列会持久化到相关性缓存里，所以重启后仍会沿完整 passing 顺序继续展示。</div>
              {factorCorrelations?.meta ? (
                <div className="mt-1 text-[11px] text-muted-foreground">
                  时序缓存 {factorCorrelations.meta.actual_trend_count ?? 0}/{factorCorrelations.meta.expected_trend_count ?? 0}
                  {factorCorrelations.meta.refreshing ? '，后台刷新中…' : ''}
                </div>
              ) : null}
            </div>
            <div className="h-[420px]">
              {correlationLoading ? (
                <div className="flex h-full items-center justify-center rounded-2xl bg-slate-50 text-xs text-muted-foreground">相关性时序加载中…</div>
              ) : correlationError ? (
                <div className="flex h-full items-center justify-center rounded-2xl bg-red-50 px-4 text-xs text-red-600">{correlationError}</div>
              ) : (
                <FactorCorrelationTrendChart factorCorrelations={factorCorrelations ?? undefined} factors={factors} />
              )}
            </div>
          </div>

          <div className="rounded-3xl border border-border/50 bg-white/90 p-4">
            <div className="mb-3 flex items-center gap-2 text-sm font-medium text-foreground">
              <FileStack className="h-4 w-4 text-sky-500" />
              最近输出文件
            </div>
            <div className="max-h-[460px] space-y-3 overflow-y-auto pr-2">
              {tableLoading && !tableKnowledge ? (
                <div className="text-sm text-muted-foreground">输出文件加载中…</div>
              ) : outputFiles.length === 0 ? (
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
              {tableLoading && !tableKnowledge ? (
                <div className="text-sm text-muted-foreground">因子卡片索引加载中…</div>
              ) : researchReports.length === 0 ? (
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

      <Panel title="OOS 区间差异" subtitle="2022-2023 只作为挖掘与门控区间；2024 OOS 和三年整体只用于展示与提交产物检查，不回流到因子发现。">
        {oosComparisonFactors.length === 0 ? (
          <div className="rounded-2xl bg-slate-50 p-4 text-sm text-muted-foreground">还没有包含 2024 OOS 指标的通过因子。</div>
        ) : (
          <div className="overflow-auto rounded-2xl border border-border/40 bg-white/80">
            <table className="min-w-[980px] text-sm">
              <thead className="bg-slate-50 text-left text-xs uppercase tracking-[0.18em] text-muted-foreground">
                <tr>
                  <th className="px-3 py-3">Factor</th>
                  <th className="px-3 py-3 text-right">Train IC</th>
                  <th className="px-3 py-3 text-right">OOS IC</th>
                  <th className="px-3 py-3 text-right">ΔIC</th>
                  <th className="px-3 py-3 text-right">Train Score</th>
                  <th className="px-3 py-3 text-right">OOS Score</th>
                  <th className="px-3 py-3 text-right">ΔScore</th>
                  <th className="px-3 py-3 text-right">Full Score</th>
                  <th className="px-3 py-3">趋势</th>
                </tr>
              </thead>
              <tbody>
                {oosComparisonFactors.map((factor) => {
                  const rows = factorPeriodRows(factor);
                  const train = rows.find((row) => row.period === 'discovery') || rows[0];
                  const oos = rows.find((row) => row.period === 'oos_2024') || rows[1];
                  const full = rows.find((row) => row.period === 'full_2022_2024') || rows[2];
                  const comparison = factor.oos_comparison || {};
                  const icDelta = Number(comparison.IC_delta ?? Number(oos?.IC || 0) - Number(train?.IC || 0));
                  const scoreDelta = Number(comparison.Score_delta ?? Number(oos?.Score || 0) - Number(train?.Score || 0));
                  return (
                    <tr key={`oos-${factor.run_id}`} className="border-t border-border/30">
                      <td className="px-3 py-3">
                        <button onClick={() => setSelectedRunId(factor.run_id)} className="break-all text-left font-mono text-xs font-medium text-sky-700 underline decoration-sky-400 underline-offset-4 hover:text-sky-900">
                          {factor.run_id}
                        </button>
                      </td>
                      <td className="px-3 py-3 text-right font-mono">{formatNumber(Number(train?.IC || 0), 3)}</td>
                      <td className="px-3 py-3 text-right font-mono">{formatNumber(Number(oos?.IC || 0), 3)}</td>
                      <td className={`px-3 py-3 text-right font-mono ${deltaTone(icDelta)}`}>{formatSignedNumber(icDelta, 3)}</td>
                      <td className="px-3 py-3 text-right font-mono">{formatNumber(Number(train?.Score || 0), 2)}</td>
                      <td className="px-3 py-3 text-right font-mono">{formatNumber(Number(oos?.Score || 0), 2)}</td>
                      <td className={`px-3 py-3 text-right font-mono ${deltaTone(scoreDelta)}`}>{formatSignedNumber(scoreDelta, 2)}</td>
                      <td className="px-3 py-3 text-right font-mono">{formatNumber(Number(full?.Score || 0), 2)}</td>
                      <td className="px-3 py-3 text-xs text-muted-foreground">{trendText(comparison)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <Panel title="知识库因子表" subtitle="每个因子共享一个名称列，并按三行展示挖掘区间、2024 OOS、三年整体指标。">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div className="text-sm text-muted-foreground">
            当前渲染 {visibleFactors.length} / {factors.length} 条记录，避免一次性挂载整张大表拖慢页面。
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setVisibleFactorCount(Math.min(factors.length, FACTOR_TABLE_BATCH_SIZE))}
              disabled={visibleFactorCount <= Math.min(factors.length, FACTOR_TABLE_BATCH_SIZE)}
              className={actionButtonClass}
            >
              前 {Math.min(factors.length, FACTOR_TABLE_BATCH_SIZE)} 条
            </button>
            <button
              onClick={() => setVisibleFactorCount((current) => Math.min(factors.length, current + FACTOR_TABLE_BATCH_SIZE))}
              disabled={visibleFactorCount >= factors.length}
              className={actionButtonClass}
            >
              再加载 {Math.min(FACTOR_TABLE_BATCH_SIZE, Math.max(factors.length - visibleFactorCount, 0))} 条
            </button>
            <button
              onClick={() => setVisibleFactorCount(factors.length)}
              disabled={visibleFactorCount >= factors.length}
              className={actionButtonClass}
            >
              全部
            </button>
          </div>
        </div>
        <div className="max-h-[720px] overflow-auto rounded-3xl border border-border/40 bg-white/70">
          <table className="min-w-[1500px] table-fixed text-sm">
            <thead className="sticky top-0 z-10 bg-white">
              <tr className="border-b border-border/50 text-left text-xs uppercase tracking-[0.18em] text-muted-foreground">
                <th className="w-16 px-3 py-3">Rank</th>
                <th className="w-36 px-3 py-3">Run ID</th>
                <th className="w-24 px-3 py-3">Window</th>
                <th className="w-20 px-3 py-3 text-right">Score</th>
                <th className="w-16 px-3 py-3 text-right">IC</th>
                <th className="w-16 px-3 py-3 text-right">IR</th>
                <th className="w-16 px-3 py-3 text-right">TVR</th>
                <th className="w-16 px-3 py-3 text-center">Days</th>
                <th className="w-24 px-3 py-3 text-center">Gen/No.</th>
                <th className="w-[30rem] px-3 py-3">Formula</th>
                <th className="w-[22rem] px-3 py-3">Thought</th>
                <th className="w-64 px-3 py-3">Status/Gate</th>
              </tr>
            </thead>
            <tbody>
              {tableLoading && !tableKnowledge ? (
                <tr>
                  <td colSpan={12} className="px-3 py-12 text-center text-sm text-muted-foreground">
                    因子表加载中…
                  </td>
                </tr>
              ) : visibleFactors.length === 0 ? (
                <tr>
                  <td colSpan={12} className="px-3 py-12 text-center text-sm text-muted-foreground">
                    {factors.length === 0 ? '还没有因子记录。启动循环后，这里会持续刷新。' : '当前没有可显示的因子记录。'}
                  </td>
                </tr>
              ) : (
                visibleFactors.map((factor) => {
                  const reason = factorFailureReason(factor);
                  const screenDetails = screenFailureDetails(factor);
                  const periodRows = factorPeriodRows(factor);
                  const rowSpan = periodRows.length;
                  return (
                    <React.Fragment key={factor.run_id}>
                      {periodRows.map((period, periodIndex) => (
                        <tr key={`${factor.run_id}-${period.period}`} className="border-b border-border/20 transition-colors" style={{ background: factorRowBackground(factor, maxScore) }}>
                          {periodIndex === 0 ? (
                            <>
                              <td rowSpan={rowSpan} className="px-3 py-3 align-top font-semibold text-foreground">#{factor.rank}</td>
                              <td rowSpan={rowSpan} className="px-3 py-3 align-top">
                                {isSubmitReady(factor) ? (
                                  <button onClick={() => setSelectedRunId(factor.run_id)} className="break-all text-left font-mono text-xs font-medium text-sky-700 underline decoration-sky-400 underline-offset-4 hover:text-sky-900">
                                    {factor.run_id}
                                  </button>
                                ) : (
                                  <div className="break-all font-mono text-xs text-foreground">{factor.run_id}</div>
                                )}
                                <div className="mt-1 text-[11px] text-muted-foreground">{formatDateTime(factor.created_at)}</div>
                              </td>
                            </>
                          ) : null}
                          <td className="px-3 py-3 align-top">
                            <div className="font-medium text-foreground">{periodLabel(period)}</div>
                            <div className="mt-1 text-[11px] text-muted-foreground">
                              {period.used_for_discovery ? '用于发现' : '只展示'}
                            </div>
                          </td>
                          <td className="px-3 py-3 text-right font-semibold text-foreground">{formatNumber(Number(period.Score || 0), 2)}</td>
                          <td className="px-3 py-3 text-right font-mono text-foreground">{formatNumber(Number(period.IC || 0), 3)}</td>
                          <td className="px-3 py-3 text-right font-mono text-foreground">{formatNumber(Number(period.IR || 0), 3)}</td>
                          <td className="px-3 py-3 text-right font-mono text-foreground">{formatNumber(Number(period.tvr ?? period.Turnover ?? 0), 0)}</td>
                          <td className="px-3 py-3 text-center text-foreground">{period.days || '-'}</td>
                          {periodIndex === 0 ? (
                            <>
                              <td rowSpan={rowSpan} className="px-3 py-3 align-top text-center text-foreground">
                                {factor.generation}/{factorOrdinalByRunId.get(factor.run_id) || '-'}
                              </td>
                              <td rowSpan={rowSpan} className="px-3 py-3 align-top">
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
                              <td rowSpan={rowSpan} className="px-3 py-3 align-top">
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
                              <td rowSpan={rowSpan} className="px-3 py-3 align-top">
                                <div className={`${pillButtonClass} max-w-full ${gateTone(reason)}`}>
                                  <span className="break-words">{reason}</span>
                                </div>
                                <div className="mt-1 text-[11px] text-muted-foreground">{statusText(factor.status)}</div>
                                {screenDetails.length ? (
                                  <div className="mt-2 space-y-1 text-[11px] leading-4 text-slate-600">
                                    {screenDetails.slice(0, 4).map((detail) => (
                                      <div key={`${factor.run_id}-${detail.key}-${detail.message || detail.value}`}>
                                        <span className="font-medium text-slate-700">{screenDetailLabel(detail)}</span>
                                        <span className="ml-1 font-mono">{screenDetailText(detail)}</span>
                                      </div>
                                    ))}
                                  </div>
                                ) : null}
                              </td>
                            </>
                          ) : null}
                        </tr>
                      ))}
                    </React.Fragment>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </Panel>

      {selectedRunId ? <ResearchModal runId={selectedRunId} onClose={() => setSelectedRunId(null)} /> : null}
      {generationNote ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 p-4 backdrop-blur-sm">
          <div className="max-h-[88vh] w-full max-w-4xl overflow-hidden rounded-3xl border border-border/70 bg-white shadow-2xl">
            <div className="flex items-start justify-between gap-4 border-b border-border/60 px-5 py-4">
              <div className="min-w-0">
                <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
                  <FileText className="h-4 w-4 text-sky-500" />
                  Generation Experience
                </div>
                <div className="mt-2 text-xl font-semibold text-foreground">Generation {generationNote.generation}</div>
                {generationNote.relative_path ? (
                  <div className="mt-1 break-all text-xs text-muted-foreground">{generationNote.relative_path}</div>
                ) : null}
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <button
                  onClick={handleRegenerateGenerationNote}
                  disabled={generationNoteRegenerating}
                  className="rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-sm text-amber-700 transition-colors hover:bg-amber-100 disabled:opacity-50"
                  title={generationNote.markdown || generationNote.summary ? '重新调用 LLM 生成经验总结' : '调用 LLM 生成经验总结'}
                >
                  {generationNoteRegenerating ? '生成中…' : (generationNote.markdown || generationNote.summary ? '重新生成' : '生成')}
                </button>
                <button onClick={() => setGenerationNote(null)} className="rounded-full border border-border/60 px-3 py-1 text-sm text-muted-foreground transition-colors hover:text-foreground">
                  关闭
                </button>
              </div>
            </div>
            <div className="max-h-[72vh] overflow-y-auto p-5">
              {generationNote.stats ? (
                <div className="mb-4 grid gap-3 sm:grid-cols-3">
                  <div className="rounded-2xl bg-slate-50 p-3 text-sm">
                    <div className="text-xs text-muted-foreground">测试数</div>
                    <div className="mt-1 text-lg font-semibold">{generationNote.stats.total}</div>
                  </div>
                  <div className="rounded-2xl bg-slate-50 p-3 text-sm">
                    <div className="text-xs text-muted-foreground">通过数</div>
                    <div className="mt-1 text-lg font-semibold">{generationNote.stats.passing}</div>
                  </div>
                  <div className="rounded-2xl bg-slate-50 p-3 text-sm">
                    <div className="text-xs text-muted-foreground">Best Score</div>
                    <div className="mt-1 text-lg font-semibold">{formatNumber(generationNote.stats.best_score, 2)}</div>
                  </div>
                </div>
              ) : null}
              {generationNote.markdown || generationNote.summary ? (
                <pre className="whitespace-pre-wrap break-words rounded-2xl bg-slate-950 p-5 font-mono text-xs leading-6 text-slate-100">
                  {generationNote.markdown || generationNote.summary}
                </pre>
              ) : (
                <div className="flex flex-col items-center gap-3 py-12 text-center text-sm text-muted-foreground">
                  <FileText className="h-8 w-8 opacity-30" />
                  <div>该 Generation 暂无经验总结。</div>
                  <div className="text-xs">点击右上角「生成」按钮，调用 LLM 生成总结；循环运行中会在每轮结束后自动生成。</div>
                </div>
              )}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};
