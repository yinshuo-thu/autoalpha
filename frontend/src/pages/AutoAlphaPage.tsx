import React, { startTransition, useEffect, useMemo, useRef, useState } from 'react';
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { BrainCircuit, FlaskConical, FolderSync, Sparkles, Wand2 } from 'lucide-react';

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
  errors?: string;
  gates_detail?: Record<string, boolean>;
  parquet_path?: string;
  research_path?: string;
  factor_card_path?: string;
  parent_run_ids?: string[];
  live_submitted?: boolean;
  live_test_result?: {
    raw?: string;
    data?: any;
    submitted_at?: string;
  };
}

interface LoopStatus {
  is_running: boolean;
  pid?: number | null;
  run_state?: {
    started_at?: string;
    params?: {
      rounds?: number;
      ideas?: number;
      days?: number;
      target_valid?: number;
      run_model_lab?: boolean;
    };
  };
  total_tested: number;
  total_passing: number;
  best_score: number;
  best_ic?: number;
  updated_at: string;
  logs: string[];
}

interface AutoAlphaFile {
  name: string;
  path: string;
  relative_path: string;
  kind: string;
  size_bytes: number;
  modified_at: string;
}

interface ProgressPoint {
  index: number;
  timestamp: string;
  label: string;
  tested: number;
  passing: number;
  best_score: number;
  best_ic: number;
}

interface GenerationSummary {
  generation: number;
  total: number;
  passing: number;
  best_score: number;
}

interface KnowledgePayload extends LoopStatus {
  pass_rate: number;
  status_breakdown: Record<string, number>;
  progress_points: ProgressPoint[];
  generation_summary: GenerationSummary[];
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

interface BalancePayload {
  total_quota: number;
  used: number;
  remaining: number;
  used_pct: number;
  remaining_pct: number;
  quota_status: 'healthy' | 'warning' | 'critical' | 'exhausted';
  total_factors: number;
  passing_factors: number;
  pass_rate: number;
  avg_cost_per_factor: number;
  avg_cost_per_valid_factor: number;
  est_total_tokens: number;
  avg_tokens_per_factor: number;
  avg_tokens_per_valid_factor: number;
  warnings: string[];
}

interface ModelLabPoint {
  date: string;
  value: number;
}

interface ModelLabModelSummary {
  avg_daily_ic: number;
  avg_daily_rank_ic: number;
  avg_daily_ic_bps?: number;       // avg_daily_ic * 100 — same scale as factor IC
  avg_daily_rank_ic_bps?: number;  // avg_daily_rank_ic * 100
  avg_ir?: number;                  // IR from daily rank-IC series
  avg_sharpe: number;
  total_pnl: number;
  gross_pnl?: number;
  total_fee?: number;
  max_drawdown: number;
  hit_ratio: number;
  avg_turnover?: number;
  // Official submission-like metrics (only on best model after submit export)
  submit_IC?: number;
  submit_IR?: number;
  submit_Score?: number;
  submit_tvr?: number;
  submit_TurnoverLocal?: number;
  submit_PassGates?: boolean;
  submit_GatesDetail?: Record<string, boolean>;
  submit_combo_daily_tvr?: number;
  method_card?: {
    name: string;
    description: string;
    weight_rule: string;
    train_inputs: string;
    validation_usage: string;
    oos_usage: string;
    leakage_guard: string;
  };
  train_val_metrics?: Record<string, {
    IC?: number;
    RankIC?: number;
    IR?: number;
    Score?: number;
    TVR?: number;
    rows?: number;
    PassGates?: boolean;
    GatesDetail?: Record<string, boolean>;
  }>;
  train_val_curve?: Array<{
    period: 'train' | 'val';
    date: string;
    mean_prediction?: number;
    mean_return?: number;
    mean_prediction_aligned?: number;
    predicted_spread: number;
    realized_spread: number;
    predicted_spread_aligned?: number;
  }>;
  combo_tvr_curve?: ModelLabPoint[];
  combo_weights?: Array<{ factor: string; weight: number }>;
  cumulative_curve: ModelLabPoint[];
  drawdown_curve?: ModelLabPoint[];
  gross_cumulative_curve?: ModelLabPoint[];
  fee_cumulative_curve?: ModelLabPoint[];
  daily_pnl_curve: ModelLabPoint[];
  long_only_cumulative_curve?: ModelLabPoint[];
  long_only_drawdown_curve?: ModelLabPoint[];
  long_only_gross_cumulative_curve?: ModelLabPoint[];
  long_only_fee_cumulative_curve?: ModelLabPoint[];
  daily_long_pnl_curve?: ModelLabPoint[];
  prediction_comparison_curve?: Array<{
    date: string;
    mean_prediction: number;
    mean_return: number;
    mean_prediction_aligned?: number;
    predicted_spread: number;
    realized_spread: number;
    predicted_spread_aligned?: number;
    daily_ic?: number;
    daily_rank_ic?: number;
    daily_pnl?: number;
    window_id?: number;
    test_start?: string;
    test_end?: string;
  }>;
  top_features: Array<{
    factor: string;
    importance: number;
  }>;
  input_factor_correlations?: Array<{
    run_id: string;
    corr: number;
    abs_corr: number;
    score?: number;
    ic?: number;
    generation?: number;
    formula?: string;
  }>;
  all_factor_correlations?: Array<{
    run_id: string;
    corr: number;
    abs_corr: number;
    score?: number;
    ic?: number;
    generation?: number;
    formula?: string;
    is_input_factor?: boolean;
  }>;
}

interface ModelLabWindow {
  window_id: number;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
  train_rows: number;
  test_rows: number;
  models: Record<
    string,
    {
      daily_ic_mean: number;
      daily_rank_ic_mean: number;
      overall_ic: number;
      rows: number;
      pnl: number;
      sharpe: number;
      max_drawdown: number;
      hit_ratio: number;
      avg_turnover?: number;
      gross_pnl?: number;
      total_fee?: number;
    }
  >;
}

interface ModelLabSummary {
  run_id: string;
  created_at: string;
  target_valid_count: number;
  selected_factor_count: number;
  window_count?: number;
  train_days?: number;
  test_days?: number;
  step_days?: number;
  train_period_start?: string;
  train_period_end?: string;
  eval_period_start?: string;
  eval_period_end?: string;
  best_ic?: number;
  best_score?: number;
  best_model: string;
  selected_factors: Array<{
    run_id: string;
    score: number;
    ic: number;
    generation: number;
    formula: string;
  }>;
  windows: ModelLabWindow[];
  models: Record<string, ModelLabModelSummary>;
  ensemble_outputs?: Record<string, string>;
  best_model_input_factor_correlations?: Array<{
    run_id: string;
    corr: number;
    abs_corr: number;
    score?: number;
    ic?: number;
    generation?: number;
    formula?: string;
  }>;
  best_model_all_factor_correlations?: Array<{
    run_id: string;
    corr: number;
    abs_corr: number;
    score?: number;
    ic?: number;
    generation?: number;
    formula?: string;
    is_input_factor?: boolean;
  }>;
  fusion_lab?: FusionLabSummary;
}

interface ModelLabPayload {
  latest: ModelLabSummary | null;
  low_corr_exploration?: ModelLabSummary | null;
  runs: Array<{
    run_id: string;
    relative_path: string;
    modified_at: string;
    target_valid_count?: number;
    selected_factor_count?: number;
    window_count?: number;
    best_model?: string;
    models?: Record<string, ModelLabModelSummary>;
  }>;
}

interface FusionLabSummary {
  created_at?: string;
  selected_model?: string;
  selected_mechanism?: string;
  best_oos_fusion_model?: string;
  best_oos_fusion_mechanism?: string;
  leakage_note?: string;
  candidate_models?: Array<{
    model: string;
    val_score: number;
    val_ic: number;
    val_ir: number;
    submit_score?: number;
    submit_ic?: number;
    submit_ir?: number;
  }>;
  output_correlation_matrix?: Array<{
    model: string;
    label: string;
    values: Array<{ model: string; label: string; corr: number }>;
  }>;
  top_output_correlation_pairs?: Array<{ left: string; right: string; corr: number; abs_corr: number }>;
  fusion_weights?: Record<string, number>;
  candidate_mechanisms?: Array<{
    name: string;
    validation_proxy: number;
    corr_penalty: number;
    selection_objective: number;
    weights: Record<string, number>;
  }>;
  fusion_results?: Array<{
    model: string;
    mechanism: string;
    Score: number;
    IC: number;
    IR: number;
    TVR: number;
    selection_objective?: number;
    weights?: Record<string, number>;
  }>;
  oos_result?: {
    model?: string;
    mechanism?: string;
    Score?: number;
    IC?: number;
    IR?: number;
    TVR?: number;
  };
}

interface InspirationRecord {
  id: number;
  kind: string;
  title: string;
  source: string;
  source_type?: string;
  content: string;
  summary: string;
  relative_path: string;
  created_at: string;
}

interface InspirationPayload {
  items: InspirationRecord[];
  count: number;
  prompt_dir: string;
  database_path: string;
  prompt_context_preview: string;
}

interface RuntimeConfigPayload {
  env?: Record<string, string>;
  paths?: Record<string, string>;
}

interface ResearchReport {
  run_id?: string;
  formula?: string;
  factor_card?: {
    run_id?: string;
    title?: string;
    status?: string;
    theme?: string;
    metrics?: Record<string, any>;
    diagnostics?: Record<string, any>;
  };
  factor_card_path?: string;
}

interface ManualFactor {
  factor_name?: string;
  formula?: string;
  recommendation?: string;
  source_mode?: string;
  Score?: number;
  IC?: number;
  IR?: number;
  Turnover?: number;
  TurnoverLocal?: number;
  PassGates?: boolean;
  submission_ready_flag?: boolean;
  sanity_report?: Record<string, any>;
  gates_detail?: Record<string, boolean>;
  submission_path?: string;
  metadata_path?: string;
}

interface EnsembleBacktestEntry {
  ic: string;
  ir: string;
  score: string;
  submitDate: string;
  notes: string;
}

const COLORS = {
  used: '#f97316',
  remaining: '#10b981',
  tested: '#60a5fa',
  passing: '#059669',
  passRate: '#7c3aed',
  efficiency: '#ea580c',
  bestScore: '#0f766e',
  score: '#15803d',
  ic: '#2563eb',
  histogram: '#7c3aed',
  error: '#ef4444',
  pnlA: '#0f766e',
  pnlB: '#2563eb',
  pnlC: '#9333ea',
  tvr: '#0f766e',
  sharpe: '#be123c',
  netPnl: '#0369a1',
};

type MethodGroupKey = 'linear' | 'ml' | 'combo';

const METHOD_GROUP_META: Record<MethodGroupKey, { rank: number; rowClass: string; textClass: string; label: string }> = {
  linear: {
    rank: 0,
    rowClass: 'bg-sky-50/70 hover:bg-sky-100/80',
    textClass: 'text-sky-700',
    label: 'Linear',
  },
  ml: {
    rank: 1,
    rowClass: 'bg-violet-50/70 hover:bg-violet-100/80',
    textClass: 'text-violet-700',
    label: 'ML / DL',
  },
  combo: {
    rank: 2,
    rowClass: 'bg-emerald-50/70 hover:bg-emerald-100/80',
    textClass: 'text-emerald-700',
    label: 'Combo',
  },
};

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const text = await res.text();
  const payload = text ? JSON.parse(text) : {};
  if (!payload.success) {
    throw new Error(payload.error || 'API error');
  }
  return payload.data as T;
}

function formatMoney(value: number) {
  return `$${value.toFixed(2)}`;
}

function formatNumber(value: number, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : '--';
}

function formatInteger(value: number) {
  return new Intl.NumberFormat('zh-CN').format(Math.round(value));
}

function formatAxisInteger(value: number | string) {
  return String(Math.round(Number(value)));
}

function formatAxisOneDecimal(value: number | string) {
  return Number(value).toFixed(1);
}

function ceilToOneDecimal(value: number) {
  return Math.ceil(value * 10) / 10;
}

function stripComboName(value: string) {
  return value.replace(/Combo/g, '');
}

function getMethodGroup(modelName: string): MethodGroupKey {
  const lower = modelName.toLowerCase();
  const isMetaModel = lower.includes('metamodel') || lower.includes('regressor') || lower.includes('forest') || lower.includes('boost');
  const isMlOrDl =
    isMetaModel
    || lower.includes('lightgbm')
    || lower.includes('xgboost')
    || lower.includes('randomforest')
    || lower.includes('extratrees')
    || lower.includes('histgradient')
    || lower.includes('mlp')
    || lower.includes('torch')
    || lower.includes('neural')
    || lower.includes('transformer')
    || lower.includes('causaldecay')
    || lower.includes('factortoken')
    || lower.includes('fusion');
  if (isMetaModel && (lower.includes('ridge') || lower.includes('linear') || lower.includes('lasso') || lower.includes('elastic'))) {
    return 'linear';
  }
  if (isMlOrDl) {
    return 'ml';
  }
  return 'combo';
}

function getMethodGroupMeta(modelName: string) {
  return METHOD_GROUP_META[getMethodGroup(modelName)];
}

function sortModelEntries(entries: Array<[string, ModelLabModelSummary]>) {
  return entries
    .filter(([modelName, payload]) => modelName !== 'EqualWeightRankCombo' || ((payload.submit_IC ?? 0) >= 0))
    .sort(([aName, aPayload], [bName, bPayload]) => {
      const scoreDiff = Number(bPayload.submit_Score ?? 0) - Number(aPayload.submit_Score ?? 0);
      if (Math.abs(scoreDiff) > 1e-9) return scoreDiff;
      const icDiff = Number(bPayload.submit_IC ?? 0) - Number(aPayload.submit_IC ?? 0);
      if (Math.abs(icDiff) > 1e-9) return icDiff;
      return aName.localeCompare(bName);
    });
}

function sortModelEntriesByGroup(entries: Array<[string, ModelLabModelSummary]>) {
  return sortModelEntries(entries).sort(([aName, aPayload], [bName, bPayload]) => {
    const aGroup = getMethodGroup(aName);
    const bGroup = getMethodGroup(bName);
    const groupDiff = METHOD_GROUP_META[aGroup].rank - METHOD_GROUP_META[bGroup].rank;
    if (groupDiff !== 0) return groupDiff;

    const scoreDiff = Number(bPayload.submit_Score ?? 0) - Number(aPayload.submit_Score ?? 0);
    if (Math.abs(scoreDiff) > 1e-9) return scoreDiff;
    const icDiff = Number(bPayload.submit_IC ?? 0) - Number(aPayload.submit_IC ?? 0);
    if (Math.abs(icDiff) > 1e-9) return icDiff;
    return aName.localeCompare(bName);
  });
}

function groupSeparators(rows: Array<{ modelLabel: string; group: MethodGroupKey }>) {
  return rows
    .map((row, index) => ({ ...row, index }))
    .filter((row, index) => index > 0 && row.group !== rows[index - 1].group);
}

function formatPercent(value: number) {
  return `${value.toFixed(1)}%`;
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

function formatAxisTime(value?: string) {
  if (!value) return '--';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return truncate(value, 10);
  return date.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function truncate(text: string, length: number) {
  if (!text) return '—';
  return text.length > length ? `${text.slice(0, length)}…` : text;
}

function compactFeatureLabel(value: string) {
  if (!value) return '—';
  return value.length <= 20 ? value : `${value.slice(0, 8)}…${value.slice(-6)}`;
}

function buildHistogram(values: number[], bins = 7, digits = 1) {
  if (values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) {
    return [{ label: min.toFixed(digits), tickLabel: min.toFixed(digits), count: values.length }];
  }
  const width = (max - min) / bins;
  const counts = Array.from({ length: bins }, (_, index) => {
    const start = min + width * index;
    const end = index === bins - 1 ? max : start + width;
    return {
      label: `${start.toFixed(digits)} ~ ${end.toFixed(digits)}`,
      tickLabel: ((start + end) / 2).toFixed(digits),
      count: 0,
    };
  });
  values.forEach((value) => {
    const rawIndex = Math.floor((value - min) / width);
    const index = Math.min(Math.max(rawIndex, 0), bins - 1);
    counts[index].count += 1;
  });
  return counts;
}

function quotaTone(status: BalancePayload['quota_status']) {
  if (status === 'healthy') return 'text-emerald-400';
  if (status === 'warning') return 'text-amber-400';
  if (status === 'critical') return 'text-orange-400';
  return 'text-red-400';
}

function niceUpperBound(value: number, minUpper: number, padding = 1.25) {
  const raw = Math.max(value * padding, minUpper);
  if (!Number.isFinite(raw) || raw <= 0) return minUpper;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalized = raw / magnitude;
  const step = normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return step * magnitude;
}

function buildZeroAlignedDomains(leftValues: number[], rightValues: number[]): { left: [number, number]; right: [number, number] } {
  const clean = (values: number[]) => values.filter((value) => Number.isFinite(value));
  const extent = (values: number[]) => {
    const vals = clean(values);
    if (!vals.length) return { min: 0, max: 1 };
    return { min: Math.min(0, ...vals), max: Math.max(0, ...vals) };
  };
  const left = extent(leftValues);
  const right = extent(rightValues);
  const belowRatio = Math.max(
    left.min < 0 ? Math.abs(left.min) / Math.max(left.max - left.min, 1e-9) : 0,
    right.min < 0 ? Math.abs(right.min) / Math.max(right.max - right.min, 1e-9) : 0,
  );
  const ratio = Math.min(0.85, Math.max(0, belowRatio));
  const expand = ({ min, max }: { min: number; max: number }): [number, number] => {
    const pos = Math.max(max, 0);
    const neg = Math.max(Math.abs(min), 0);
    if (ratio <= 1e-6) return [0, Math.ceil(niceUpperBound(pos, 1))];
    if (ratio >= 0.999) return [-Math.ceil(niceUpperBound(neg, 1)), 0];
    const upperFromNeg = neg * (1 - ratio) / ratio;
    const lowerFromPos = pos * ratio / (1 - ratio);
    const upper = Math.max(pos, upperFromNeg, 1);
    const lower = Math.max(neg, lowerFromPos, 1);
    return [Math.floor(-lower * 1.12), Math.ceil(upper * 1.12)];
  };
  return { left: expand(left), right: expand(right) };
}

type PnlDrawdownPoint = {
  date: string;
  pnl: number;
  drawdown: number;
};

function buildPnlDrawdownSeries(cumulative?: ModelLabPoint[], drawdown?: ModelLabPoint[]): PnlDrawdownPoint[] {
  const drawdownByDate = new Map((drawdown || []).map((point) => [point.date, Number(point.value) || 0]));
  return (cumulative || [])
    .map((point) => ({
      date: point.date,
      pnl: Number(point.value) || 0,
      drawdown: drawdownByDate.get(point.date) ?? 0,
    }))
    .filter((point) => Number.isFinite(point.pnl) && Number.isFinite(point.drawdown));
}

const PnlDrawdownChart = ({
  title,
  data,
  pnlColor = '#2563eb',
  heightClass = 'h-[210px]',
}: {
  title: string;
  data: PnlDrawdownPoint[];
  pnlColor?: string;
  heightClass?: string;
}) => {
  const finalPnl = data.length ? data[data.length - 1].pnl : 0;
  const maxDrawdown = data.length ? Math.min(...data.map((point) => point.drawdown)) : 0;
  return (
    <div className="rounded-2xl bg-slate-50 p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs font-medium text-slate-700">{title}</div>
        <div className="flex gap-2 text-[11px] text-slate-600">
          <span>PnL {formatNumber(finalPnl, 4)}</span>
          <span>Max DD {formatNumber(maxDrawdown, 4)}</span>
        </div>
      </div>
      {data.length ? (
        <div className={heightClass}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={formatAxisTime} minTickGap={26} />
              <YAxis yAxisId="left" tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={(value) => formatNumber(Number(value), 3)} />
              <YAxis yAxisId="right" orientation="right" tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={(value) => formatNumber(Number(value), 3)} />
              <Tooltip formatter={(value: number) => formatNumber(Number(value), 5)} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <ReferenceLine yAxisId="left" y={0} stroke="#94a3b8" strokeDasharray="3 3" />
              <ReferenceLine yAxisId="right" y={0} stroke="#94a3b8" strokeDasharray="3 3" />
              <Line yAxisId="left" type="linear" dataKey="pnl" name="累计 PnL" stroke={pnlColor} strokeWidth={2.3} dot={false} />
              <Line yAxisId="right" type="linear" dataKey="drawdown" name="Max DD" stroke="#ef4444" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="py-8 text-sm text-muted-foreground">暂无 PnL / Max DD 曲线。</div>
      )}
    </div>
  );
};

function buildAbsCorrelationHistogram(
  rows: Array<{ abs_corr?: number }>,
  binCount = 10,
) {
  if (!rows.length) return [];
  const bins = Array.from({ length: binCount }, (_, index) => ({
    bucket: `${(index / binCount).toFixed(1)}-${((index + 1) / binCount).toFixed(1)}`,
    count: 0,
    inputCount: 0,
  }));
  rows.forEach((row: any) => {
    const value = Math.max(0, Math.min(0.999999, Number(row.abs_corr) || 0));
    const index = Math.min(binCount - 1, Math.floor(value * binCount));
    bins[index].count += 1;
    if (row.is_input_factor) bins[index].inputCount += 1;
  });
  return bins;
}

function getComboDisplayTvr(payload?: ModelLabModelSummary | null) {
  if (!payload) return 0;
  const explicitDaily = Number(payload.submit_combo_daily_tvr);
  if (Number.isFinite(explicitDaily) && explicitDaily > 0) return explicitDaily;
  const curveValues = (payload.combo_tvr_curve || [])
    .map((point) => Number(point.value))
    .filter((value) => Number.isFinite(value));
  if (curveValues.length) {
    return curveValues.reduce((sum, value) => sum + value, 0) / curveValues.length;
  }
  const local = Number(payload.submit_TurnoverLocal);
  if (Number.isFinite(local) && local > 0) return local;
  const submit = Number(payload.submit_tvr);
  return Number.isFinite(submit) ? submit : 0;
}

function corrCellColor(value: number) {
  const clipped = Math.max(0, Math.min(1, Math.abs(Number(value) || 0)));
  const start = { r: 37, g: 99, b: 235 };
  const end = { r: 220, g: 38, b: 38 };
  const r = Math.round(start.r + (end.r - start.r) * clipped);
  const g = Math.round(start.g + (end.g - start.g) * clipped);
  const b = Math.round(start.b + (end.b - start.b) * clipped);
  return `rgb(${r}, ${g}, ${b})`;
}

function getPhaseMetric(
  payload: ModelLabModelSummary,
  phase: 'train' | 'val' | 'oos',
  key: 'Score' | 'IC',
) {
  const value = Number(payload.train_val_metrics?.[phase]?.[key]);
  return Number.isFinite(value) ? value : 0;
}

function suggestTargetValid(passingCount: number) {
  if (!Number.isFinite(passingCount) || passingCount <= 0) return 0;
  return Math.max(passingCount + 1, Math.ceil(passingCount * 1.5));
}

function compressProgressPoints(points: ProgressPoint[], maxPoints = 72) {
  if (points.length <= maxPoints) return points;
  if (maxPoints <= 1) return [points[points.length - 1]];

  const lastIndex = points.length - 1;
  const indices = new Set<number>([0, lastIndex]);
  for (let index = 0; index < maxPoints; index += 1) {
    indices.add(Math.round((lastIndex * index) / (maxPoints - 1)));
  }
  return Array.from(indices)
    .sort((a, b) => a - b)
    .map((index) => points[index]);
}

function buildProgressPointsFromFactors(factors: KbFactor[]) {
  const ordered = [...factors].sort((a, b) => {
    const aTime = new Date(a.created_at || '').getTime();
    const bTime = new Date(b.created_at || '').getTime();
    const aKey = Number.isNaN(aTime) ? Number.POSITIVE_INFINITY : aTime;
    const bKey = Number.isNaN(bTime) ? Number.POSITIVE_INFINITY : bTime;
    if (aKey !== bKey) return aKey - bKey;
    return String(a.run_id).localeCompare(String(b.run_id));
  });

  let passing = 0;
  let bestScore = 0;
  let bestIc = 0;
  const points = ordered.map((factor, index) => {
    if (factor.PassGates) passing += 1;
    bestScore = Math.max(bestScore, Number(factor.Score || 0));
    bestIc = Math.max(bestIc, Number(factor.IC || 0));
    const timestamp = factor.created_at || '';
    return {
      index: index + 1,
      timestamp,
      label: timestamp ? timestamp.replace('T', ' ').slice(0, 16) : `#${index + 1}`,
      tested: index + 1,
      passing,
      best_score: Number(bestScore.toFixed(2)),
      best_ic: Number(bestIc.toFixed(4)),
    };
  });

  return compressProgressPoints(points);
}

function resolveProgressPoints(
  apiPoints: ProgressPoint[],
  factors: KbFactor[],
  status: LoopStatus | null,
  knowledge: KnowledgePayload | null
) {
  const factorPoints = buildProgressPointsFromFactors(factors);
  if (factorPoints.length === 0) return apiPoints;

  const apiLast = apiPoints[apiPoints.length - 1];
  const factorLast = factorPoints[factorPoints.length - 1];
  const expectedTested = Math.max(status?.total_tested ?? 0, knowledge?.total_tested ?? 0, factors.length);
  const expectedPassing = Math.max(status?.total_passing ?? 0, knowledge?.total_passing ?? 0, factorLast?.passing ?? 0);
  const expectedBestScore = Math.max(status?.best_score ?? 0, knowledge?.best_score ?? 0, factorLast?.best_score ?? 0);
  const apiHasBestIc = apiPoints.every((point) => typeof point.best_ic === 'number');

  const apiIsCurrent =
    apiLast &&
    apiHasBestIc &&
    apiLast.tested >= expectedTested &&
    apiLast.passing >= expectedPassing &&
    apiLast.best_score >= Number(expectedBestScore.toFixed(2));

  return apiIsCurrent ? apiPoints : factorPoints;
}

const Panel = ({
  title,
  subtitle,
  right,
  children,
  className = '',
}: {
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) => (
  <section className={`glass min-w-0 max-w-full overflow-hidden rounded-[28px] border border-border/60 p-4 sm:p-5 ${className}`}>
    <div className="mb-4 flex min-w-0 flex-wrap items-start justify-between gap-4">
      <div className="min-w-0">
        <div className="break-words text-[11px] uppercase tracking-[0.22em] text-muted-foreground">{title}</div>
        {subtitle ? <div className="mt-2 max-w-full break-words text-sm leading-6 text-muted-foreground">{subtitle}</div> : null}
      </div>
      {right ? <div className="shrink-0">{right}</div> : null}
    </div>
    {children}
  </section>
);

const StatCard = ({
  label,
  value,
  helper,
  accent = '',
  valueClassName = '',
}: {
  label: string;
  value: string;
  helper?: string;
  accent?: string;
  valueClassName?: string;
}) => (
  <div className={`min-w-0 max-w-full overflow-hidden rounded-3xl border border-border/50 bg-white/80 p-4 shadow-sm ${accent}`}>
    <div className="text-xs text-muted-foreground">{label}</div>
    <div className={`mt-3 min-w-0 break-words text-3xl font-semibold leading-tight tracking-tight text-foreground ${valueClassName}`}>{value}</div>
    {helper ? <div className="mt-2 break-words text-xs leading-5 text-muted-foreground">{helper}</div> : null}
  </div>
);

class PageErrorBoundary extends React.Component<{ children: React.ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div className="rounded-3xl border border-red-200 bg-red-50 p-5 text-sm leading-6 text-red-700">
          <div className="font-semibold">前端图表渲染异常</div>
          <div className="mt-2 break-words">{this.state.error.message || '未知错误'}</div>
          <div className="mt-2 text-xs text-red-600">页面数据已保留，刷新后会重新拉取最新快照。</div>
        </div>
      );
    }
    return this.props.children;
  }
}

const LogPanel = ({ logs }: { logs: string[] }) => {
  const ref = useRef<HTMLDivElement>(null);
  const taggedSegments = (line: string) => line.split(/(\[(?:Recall|Embeding|Embedding)\])/g).filter(Boolean);
  const isRecallTag = (part: string) => part === '[Recall]';
  const isEmbeddingTag = (part: string) => part === '[Embeding]' || part === '[Embedding]';

  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [logs]);

  const colorLine = (line: string) => {
    if (line.includes('[Recall]')) return 'text-cyan-300';
    if (line.includes('[Embeding]') || line.includes('[Embedding]')) return 'text-fuchsia-300';
    if (line.includes('ERROR') || line.includes('[ERROR]')) return 'text-red-500';
    if (line.includes('WARN') || line.includes('[WARN]')) return 'text-amber-500';
    if (line.includes('PassGates=True') || line.includes('LOOP COMPLETE') || line.includes('passing=')) {
      return 'text-emerald-500';
    }
    if (line.includes('ROUND') || line.includes('---')) return 'text-sky-500 font-semibold';
    return 'text-slate-300';
  };

  return (
    <div ref={ref} className="h-[320px] min-w-0 max-w-full overflow-y-auto overflow-x-hidden rounded-3xl border border-slate-800 bg-slate-950/95 p-4 font-mono text-xs shadow-inner">
      {logs.length === 0 ? (
        <div className="text-slate-500">等待挖掘日志...</div>
      ) : (
        logs.map((line, index) => (
          <div key={`${line}-${index}`} className={`mb-1 whitespace-pre-wrap break-words leading-5 [overflow-wrap:anywhere] ${colorLine(line)}`}>
            {taggedSegments(line).map((part, partIndex) => {
              if (isRecallTag(part)) {
                return (
                  <span
                    key={`${index}-${partIndex}-${part}`}
                    className="mr-2 inline-flex rounded-full border border-cyan-400/40 bg-cyan-400/15 px-2 py-0.5 text-[10px] font-semibold tracking-[0.12em] text-cyan-200"
                  >
                    [Recall]
                  </span>
                );
              }
              if (isEmbeddingTag(part)) {
                return (
                  <span
                    key={`${index}-${partIndex}-${part}`}
                    className="mr-2 inline-flex rounded-full border border-fuchsia-400/40 bg-fuchsia-400/15 px-2 py-0.5 text-[10px] font-semibold tracking-[0.12em] text-fuchsia-200"
                  >
                    [Embeding]
                  </span>
                );
              }
              return <span key={`${index}-${partIndex}`}>{part}</span>;
            })}
          </div>
        ))
      )}
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
  const metrics = card?.metrics || {};
  const diagnostics = card?.diagnostics || {};

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4" onClick={onClose}>
      <div className="max-h-[88vh] w-full max-w-3xl overflow-y-auto rounded-[28px] border border-border/50 bg-white p-6 shadow-2xl" onClick={(event) => event.stopPropagation()}>
        <div className="mb-5 flex items-center justify-between gap-3">
          <div>
            <div className="text-[11px] uppercase tracking-[0.22em] text-muted-foreground">Factor Card</div>
            <div className="mt-2 text-xl font-semibold text-foreground">{runId}</div>
          </div>
          <button onClick={onClose} className="rounded-full border border-border/60 px-3 py-1 text-xs text-foreground transition-colors hover:bg-slate-50">
            关闭
          </button>
        </div>
        {error ? <div className="rounded-2xl bg-red-50 p-4 text-sm text-red-600">{error}</div> : null}
        {!report && !error ? <div className="text-sm text-muted-foreground">Factor Card 加载中...</div> : null}
        {report ? (
          <div className="space-y-4">
            <div className="rounded-3xl border border-border/50 bg-slate-50 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="text-lg font-semibold text-foreground">{card?.title || card?.run_id || runId}</div>
                  {card?.theme ? <div className="mt-1 text-sm text-muted-foreground">{card.theme}</div> : null}
                </div>
                {card?.status ? (
                  <div className={`rounded-full px-3 py-1 text-xs font-semibold ${card.status === 'PASS' ? 'bg-emerald-600 text-white' : 'bg-slate-200 text-slate-700'}`}>
                    {card.status}
                  </div>
                ) : null}
              </div>
              <div className="mt-4 grid gap-3 md:grid-cols-4">
                {[
                  ['Score', metrics.Score],
                  ['IC', metrics.IC],
                  ['IR', metrics.IR],
                  ['TVR', metrics.tvr ?? metrics.TVR],
                ].map(([label, value]) => (
                  <div key={String(label)} className="rounded-2xl bg-white p-3">
                    <div className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">{label}</div>
                    <div className="mt-2 text-lg font-semibold text-foreground">{formatNumber(Number(value || 0), label === 'TVR' ? 1 : 3)}</div>
                  </div>
                ))}
              </div>
            </div>
            {report.formula ? (
              <div className="rounded-3xl border border-border/50 bg-slate-50 p-4">
                <div className="text-sm font-medium text-foreground">公式</div>
                <div className="mt-2 break-all font-mono text-xs leading-6 text-foreground">{report.formula}</div>
              </div>
            ) : null}
            {Object.keys(diagnostics).length ? (
              <div className="rounded-3xl border border-border/50 bg-slate-50 p-4">
                <div className="text-sm font-medium text-foreground">诊断摘要</div>
                <div className="mt-3 grid gap-3 md:grid-cols-2">
                  {Object.entries(diagnostics).slice(0, 8).map(([key, value]) => (
                    <div key={key} className="rounded-2xl bg-white p-3">
                      <div className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">{key}</div>
                      <div className="mt-2 break-words text-sm text-foreground">{typeof value === 'object' ? JSON.stringify(value) : String(value)}</div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
};

const EnsembleModal = ({
  modelName,
  modelLab,
  onOpenFactor,
  onClose,
}: {
  modelName: string;
  modelLab: ModelLabSummary;
  onOpenFactor: (runId: string) => void;
  onClose: () => void;
}) => {
  const [backtest, setBacktest] = useState<EnsembleBacktestEntry>(() => {
    try { return JSON.parse(localStorage.getItem('ensemble_backtest') || '{}')[modelName] || { ic: '', ir: '', score: '', submitDate: '', notes: '' }; }
    catch { return { ic: '', ir: '', score: '', submitDate: '', notes: '' }; }
  });
  const [saved, setSaved] = useState(false);
  const modelPayload = modelLab.models?.[modelName];
  const path = modelLab.ensemble_outputs?.[modelName] || '';
  const isBest = modelName === modelLab.best_model;
  const inputCorrelations = (modelPayload?.input_factor_correlations || []).slice(0, 10);
  const allFactorCorrelations = (
    modelPayload?.all_factor_correlations
    || (isBest ? modelLab.best_model_all_factor_correlations : [])
    || []
  );
  const topAllFactorCorrelations = allFactorCorrelations.slice(0, 12);
  const methodCard = modelPayload?.method_card;
  const periodMetricRows = ['train', 'val', 'oos'].map((period) => ({
    period,
    ...(modelPayload?.train_val_metrics?.[period] || {}),
  }));
  const trainValCurve = (modelPayload?.train_val_curve || []).map((row) => ({
    date: row.date,
    period: row.period,
    predicted: row.predicted_spread_aligned ?? row.predicted_spread,
    realized: row.realized_spread,
  }));
  const allFactorCorrelationHistogram = buildAbsCorrelationHistogram(allFactorCorrelations);
  const allFactorCorrelationStats = useMemo(() => {
    if (!allFactorCorrelations.length) return null;
    const values = allFactorCorrelations.map((item) => Number(item.abs_corr || 0)).sort((a, b) => a - b);
    const quantile = (q: number) => {
      if (!values.length) return 0;
      const index = Math.min(values.length - 1, Math.max(0, Math.floor((values.length - 1) * q)));
      return values[index];
    };
    return {
      count: values.length,
      max: values[values.length - 1] || 0,
      median: quantile(0.5),
      p90: quantile(0.9),
      over07: allFactorCorrelations.filter((item) => Number(item.abs_corr || 0) >= 0.7).length,
      over05: allFactorCorrelations.filter((item) => Number(item.abs_corr || 0) >= 0.5).length,
    };
  }, [allFactorCorrelations]);

  const saveBacktest = () => {
    try {
      const all = JSON.parse(localStorage.getItem('ensemble_backtest') || '{}');
      all[modelName] = backtest;
      localStorage.setItem('ensemble_backtest', JSON.stringify(all));
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch {}
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="max-h-[90vh] w-full max-w-4xl overflow-y-auto rounded-3xl border border-border/60 bg-white p-6 shadow-2xl">
        <div className="mb-5 flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">整体合成因子</div>
            <div className="mt-1 truncate font-semibold text-foreground">{modelName}</div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {isBest && <span className="rounded-full bg-emerald-100 px-3 py-1 text-[11px] font-medium text-emerald-700">BEST</span>}
            <button onClick={onClose} className="rounded-full p-1 text-muted-foreground hover:bg-slate-100 hover:text-foreground">✕</button>
          </div>
        </div>

        {path ? (
          <div className="mb-4 rounded-2xl bg-slate-50 p-3">
            <div className="mb-1 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">输出文件路径</div>
            <div className="break-all font-mono text-[11px] leading-6 text-slate-700">{path}</div>
          </div>
        ) : null}

        {methodCard ? (
          <div className="mb-4 rounded-2xl bg-slate-50 p-3">
            <div className="mb-1 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Combo Method</div>
            <div className="text-sm font-semibold text-foreground">{methodCard.name}</div>
            <div className="mt-2 text-xs leading-6 text-slate-700">{methodCard.description}</div>
            <div className="mt-3 grid gap-2 text-[11px] leading-5 text-slate-600">
              <div className="rounded-xl bg-white p-2">权重规则：{methodCard.weight_rule}</div>
              <div className="rounded-xl bg-white p-2">验证使用：{methodCard.validation_usage}</div>
              <div className="rounded-xl bg-white p-2">泄漏检查：{methodCard.leakage_guard}</div>
            </div>
          </div>
        ) : null}

        {modelPayload?.train_val_metrics ? (
          <div className="mb-4 rounded-2xl bg-slate-50 p-3">
            <div className="mb-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Train / Val / 2024 OOS</div>
            <div className="grid gap-2 text-[11px] sm:grid-cols-3">
              {periodMetricRows.map((row) => (
                <div key={row.period} className="rounded-xl bg-white p-2">
                  <div className="font-semibold uppercase text-slate-700">{row.period}</div>
                  <div className="mt-1 text-slate-600">Score {formatNumber(Number(row.Score ?? 0), 2)}</div>
                  <div className="text-slate-600">IC {formatNumber(Number(row.IC ?? 0), 2)}</div>
                  <div className="text-slate-600">IR {formatNumber(Number(row.IR ?? 0), 2)}</div>
                  <div className="text-slate-600">TVR {formatNumber(Number(row.TVR ?? 0), 1)}</div>
                </div>
              ))}
            </div>
            {trainValCurve.length ? (
              <div className="mt-3 h-[220px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trainValCurve} margin={{ top: 8, right: 14, bottom: 8, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={formatAxisTime} minTickGap={24} />
                    <YAxis tick={{ fill: '#64748b', fontSize: 10 }} />
                    <Tooltip formatter={(value: number) => formatNumber(Number(value), 5)} />
                    <Legend />
                    <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
                    <Line type="linear" dataKey="predicted" name="Train/Val predicted spread" stroke="#2563eb" strokeWidth={2} dot={{ r: 1.4 }} />
                    <Line type="linear" dataKey="realized" name="Train/Val realized spread" stroke="#0f766e" strokeWidth={2} dot={{ r: 1.4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : null}
          </div>
        ) : null}

        {modelPayload ? (
          <div className="mb-4 rounded-2xl bg-slate-50 p-3">
            {modelPayload.submit_IC !== undefined ? (
              <>
                <div className="mb-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">官方评测指标（submission-like）</div>
                <div className="grid grid-cols-2 gap-y-2 text-[11px]">
                  <div className="font-semibold text-emerald-700">IC {formatNumber(modelPayload.submit_IC, 2)}</div>
                  <div className="font-semibold text-emerald-700">Score {formatNumber(modelPayload.submit_Score!, 2)}</div>
                  <div className="font-semibold text-emerald-700">IR {formatNumber(modelPayload.submit_IR!, 2)}</div>
	                  <div className="font-semibold text-emerald-700">TVR {formatNumber(getComboDisplayTvr(modelPayload), 1)}</div>
	                  <div className="text-slate-600">提交 TVR {formatNumber(modelPayload.submit_tvr ?? 0, 1)}</div>
                  <div className={modelPayload.submit_PassGates ? 'text-emerald-700' : 'text-red-600'}>
                    {modelPayload.submit_PassGates ? '✓ PassGates' : '✗ 未过门槛'}
                  </div>
                </div>
                <div className="mt-3 border-t border-slate-200 pt-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">2024 OOS 估算值</div>
              </>
            ) : (
              <div className="mb-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">模型表现（2024 OOS 均值 · IC 已×100对齐因子口径）</div>
            )}
            <div className="mt-1.5 grid grid-cols-3 gap-y-2 text-[11px] text-slate-600">
              <div>IC {formatNumber((modelPayload.avg_daily_rank_ic_bps ?? modelPayload.avg_daily_rank_ic * 100), 2)}</div>
              <div>IR {formatNumber(modelPayload.avg_ir ?? 0, 2)}</div>
              <div>Sharpe {formatNumber(modelPayload.avg_sharpe, 2)}</div>
              <div>TVR策略 {formatNumber(modelPayload.avg_turnover ?? 0, 3)}</div>
              <div>MaxDD {formatNumber(modelPayload.max_drawdown, 3)}</div>
              <div>Hit {formatPercent(modelPayload.hit_ratio * 100)}</div>
            </div>
          </div>
        ) : null}

        {inputCorrelations.length > 0 ? (
          <div className="mb-4 rounded-2xl bg-slate-50 p-3">
            <div className="mb-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              {isBest ? '最佳模型因子 vs 入模因子相关性分布' : '模型因子 vs 入模因子相关性分布'}
            </div>
            <div className="mb-3 text-[11px] leading-5 text-slate-500">
              这里展示的是当前模型最终输出信号与各入模原始因子的相关性，不再计算入模因子之间的两两相关。
            </div>
            <div className="h-[220px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={[...inputCorrelations].reverse()} layout="vertical" margin={{ top: 4, right: 18, bottom: 4, left: 6 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" domain={[-1, 1]} tick={{ fill: '#64748b', fontSize: 10 }} />
                  <YAxis type="category" dataKey="run_id" width={108} tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={compactFeatureLabel} />
                  <Tooltip
                    formatter={(value: number) => [formatNumber(Number(value), 3), '相关性']}
                    labelFormatter={(label, payload) => String((payload?.[0]?.payload as { run_id?: string } | undefined)?.run_id || label)}
                  />
                  <Bar dataKey="corr" radius={[0, 6, 6, 0]}>
                    {[...inputCorrelations].reverse().map((item) => (
                      <Cell key={item.run_id} fill={item.corr >= 0 ? '#0ea5e9' : '#f97316'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-3 space-y-1.5">
              {inputCorrelations.slice(0, 6).map((item) => (
                <div key={item.run_id} className="flex items-center gap-2 text-[11px]">
                  <button
                    onClick={() => { onOpenFactor(item.run_id); onClose(); }}
                    className="min-w-0 truncate font-mono text-sky-700 underline decoration-sky-400 underline-offset-4 hover:text-sky-900"
                  >
                    {item.run_id}
                  </button>
                  <span className="ml-auto shrink-0 font-semibold text-slate-700">{formatNumber(item.corr, 3)}</span>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {allFactorCorrelations.length > 0 ? (
          <div className="mb-4 rounded-2xl bg-slate-50 p-3">
            <div className="mb-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              pq 文件 vs 其他有效因子相关性分布
            </div>
            <div className="mb-3 text-[11px] leading-5 text-slate-500">
              这里展示导出的整体合成 pq 文件与当前全部有效因子的相关性分布。蓝色柱表示该区间内全部有效因子数，青绿色叠层表示其中属于当前入模集合的因子。
            </div>
            {allFactorCorrelationStats ? (
              <div className="mb-3 grid grid-cols-3 gap-2 text-[11px] text-slate-600">
                <div>样本数 {allFactorCorrelationStats.count}</div>
                <div>中位 |corr| {formatNumber(allFactorCorrelationStats.median, 3)}</div>
                <div>P90 |corr| {formatNumber(allFactorCorrelationStats.p90, 3)}</div>
                <div>最大 |corr| {formatNumber(allFactorCorrelationStats.max, 3)}</div>
                <div>|corr| ≥ 0.5: {allFactorCorrelationStats.over05}</div>
                <div>|corr| ≥ 0.7: {allFactorCorrelationStats.over07}</div>
              </div>
            ) : null}
            <div className="h-[220px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={allFactorCorrelationHistogram} margin={{ top: 4, right: 12, bottom: 10, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis dataKey="bucket" tick={{ fill: '#64748b', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#64748b', fontSize: 10 }} allowDecimals={false} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Bar dataKey="count" name="全部有效因子" fill="#60a5fa" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="inputCount" name="其中入模因子" fill="#14b8a6" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-3 space-y-1.5">
              {topAllFactorCorrelations.slice(0, 8).map((item) => (
                <div key={item.run_id} className="flex items-center gap-2 text-[11px]">
                  <button
                    onClick={() => { onOpenFactor(item.run_id); onClose(); }}
                    className="min-w-0 truncate font-mono text-sky-700 underline decoration-sky-400 underline-offset-4 hover:text-sky-900"
                  >
                    {item.run_id}
                  </button>
                  {item.is_input_factor ? (
                    <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] text-emerald-700">input</span>
                  ) : null}
                  <span className="ml-auto shrink-0 font-semibold text-slate-700">{formatNumber(item.corr, 3)}</span>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        <div className="rounded-2xl bg-slate-50 p-3">
          <div className="mb-3 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">填入回测结果</div>
          <div className="mb-2 grid grid-cols-3 gap-2">
            {(['ic', 'ir', 'score'] as const).map((field) => (
              <div key={field}>
                <label className="text-[10px] uppercase text-muted-foreground">{field.toUpperCase()}</label>
                <input
                  value={backtest[field]}
                  onChange={(e) => setBacktest((prev) => ({ ...prev, [field]: e.target.value }))}
                  placeholder="--"
                  className="mt-1 w-full rounded-xl border border-border/50 bg-white px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-sky-400"
                />
              </div>
            ))}
          </div>
          <div className="mb-2">
            <label className="text-[10px] uppercase text-muted-foreground">提交日期</label>
            <input
              type="date"
              value={backtest.submitDate}
              onChange={(e) => setBacktest((prev) => ({ ...prev, submitDate: e.target.value }))}
              className="mt-1 w-full rounded-xl border border-border/50 bg-white px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-sky-400"
            />
          </div>
          <div className="mb-3">
            <label className="text-[10px] uppercase text-muted-foreground">备注</label>
            <textarea
              value={backtest.notes}
              onChange={(e) => setBacktest((prev) => ({ ...prev, notes: e.target.value }))}
              placeholder="提交平台结果、竞赛排名等..."
              rows={2}
              className="mt-1 w-full rounded-xl border border-border/50 bg-white px-2 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-sky-400"
            />
          </div>
          <button
            onClick={saveBacktest}
            className={`w-full rounded-2xl py-2 text-xs font-medium text-white transition-colors ${saved ? 'bg-emerald-600' : 'bg-sky-600 hover:bg-sky-700'}`}
          >
            {saved ? '已保存 ✓' : '保存回测结果'}
          </button>
        </div>
      </div>
    </div>
  );
};

export const AutoAlphaPage: React.FC = () => {
  const [status, setStatus] = useState<LoopStatus | null>(null);
  const [knowledge, setKnowledge] = useState<KnowledgePayload | null>(null);
  const [balance, setBalance] = useState<BalancePayload | null>(null);
  const [modelLab, setModelLab] = useState<ModelLabPayload | null>(null);
  const [inspirations, setInspirations] = useState<InspirationPayload | null>(null);
  const [comboModal, setComboModal] = useState<{ modelName: string; lab: 'full' | 'low' } | null>(null);
  const [rounds, setRounds] = useState(10);
  const [ideas, setIdeas] = useState(4);
  const [days, setDays] = useState(0);
  const [targetValid, setTargetValid] = useState(0);
  const [targetValidSource, setTargetValidSource] = useState<'auto' | 'config' | 'manual'>('auto');
  const [promptTitle, setPromptTitle] = useState('');
  const [promptInput, setPromptInput] = useState('');
  const [promptBusy, setPromptBusy] = useState(false);
  const [manualBusy, setManualBusy] = useState(false);
  const [manualFactor, setManualFactor] = useState<ManualFactor | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [pageMessage, setPageMessage] = useState('');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const secondaryPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let mounted = true;
    let coreInflight = false;
    let secondaryInflight = false;

    const loadCore = async () => {
      if (coreInflight || document.hidden) return;
      coreInflight = true;
      try {
        const [statusResult, knowledgeResult] = await Promise.allSettled([
          fetchJson<LoopStatus>('/api/autoalpha/loop/status'),
          fetchJson<KnowledgePayload>('/api/autoalpha/knowledge?compact_factors=1&include_factor_correlations=0&include_artifacts=0&include_generation_experiences=0'),
        ]);
        if (!mounted) return;
        startTransition(() => {
          if (statusResult.status === 'fulfilled') setStatus(statusResult.value);
          if (knowledgeResult.status === 'fulfilled') setKnowledge(knowledgeResult.value);
        });
      } finally {
        coreInflight = false;
      }
    };

    const loadSecondary = async () => {
      if (secondaryInflight || document.hidden) return;
      secondaryInflight = true;
      try {
        const updates = [
          fetchJson<BalancePayload>('/api/autoalpha/balance').then((value) => {
            if (!mounted) return;
            startTransition(() => setBalance(value));
          }),
          fetchJson<ModelLabPayload>('/api/autoalpha/model-lab').then((value) => {
            if (!mounted) return;
            startTransition(() => setModelLab(value));
          }),
          fetchJson<InspirationPayload>('/api/autoalpha/inspirations').then((value) => {
            if (!mounted) return;
            startTransition(() => setInspirations(value));
          }),
        ];
        await Promise.allSettled(updates);
      } finally {
        secondaryInflight = false;
      }
    };

    const handleVisibilityChange = () => {
      if (!document.hidden) {
        void loadCore();
        void loadSecondary();
      }
    };

    void loadCore();
    void loadSecondary();
    pollRef.current = setInterval(loadCore, 6000);
    secondaryPollRef.current = setInterval(loadSecondary, 20000);
    document.addEventListener('visibilitychange', handleVisibilityChange);

    fetchJson<RuntimeConfigPayload>('/api/system/config')
      .then((cfg) => {
        const env = cfg.env || {};
        setRounds(Number(env.AUTOALPHA_DEFAULT_ROUNDS || 10));
        setIdeas(Number(env.AUTOALPHA_DEFAULT_IDEAS || 4));
        setDays(Number(env.AUTOALPHA_DEFAULT_DAYS || 0));
        const configuredTarget = Math.max(
          0,
          Number(env.AUTOALPHA_DEFAULT_TARGET_VALID || env.AUTOALPHA_ROLLING_TARGET_VALID || 0)
        );
        if (configuredTarget > 0) {
          setTargetValid(configuredTarget);
          setTargetValidSource('config');
        }
      })
      .catch(() => {});

    return () => {
      mounted = false;
      if (pollRef.current) clearInterval(pollRef.current);
      if (secondaryPollRef.current) clearInterval(secondaryPollRef.current);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  const isRunning = status?.is_running ?? false;
  const factors = knowledge?.factors ?? [];
  const currentPassingCount = Math.max(
    status?.total_passing ?? 0,
    knowledge?.total_passing ?? 0,
    factors.filter((factor) => factor.PassGates).length
  );
  const suggestedTargetValid = useMemo(
    () => suggestTargetValid(currentPassingCount),
    [currentPassingCount]
  );

  useEffect(() => {
    if (targetValidSource !== 'auto') return;
    setTargetValid(suggestedTargetValid);
  }, [suggestedTargetValid, targetValidSource]);

  const progressPoints = useMemo(
    () => resolveProgressPoints(knowledge?.progress_points ?? [], factors, status, knowledge),
    [knowledge, factors, status]
  );
  const progressChartPoints = useMemo(
    () => {
      const firstTs = new Date(progressPoints[0]?.timestamp || '').getTime();
      return progressPoints.map((point) => {
        const currentTs = new Date(point.timestamp || '').getTime();
        const elapsedHours =
          Number.isFinite(firstTs) && Number.isFinite(currentTs) && currentTs > firstTs
            ? (currentTs - firstTs) / 3_600_000
            : 0;
        return {
        ...point,
        pass_rate: point.tested > 0 ? Number(((point.passing / point.tested) * 100).toFixed(2)) : 0,
          generation_efficiency: elapsedHours > 0 ? Number((point.passing / elapsedHours).toFixed(2)) : 0,
        };
      });
    },
    [progressPoints]
  );
  const maxPassingCount = Math.max(...progressChartPoints.map((point) => point.passing), 0);
  const passingAxisMax = Math.max(maxPassingCount > 0 ? Math.ceil(maxPassingCount * 1.5) : 0, 10);
  const maxPassRate = Math.max(...progressChartPoints.map((point) => point.pass_rate), 0);
  const passRateAxisMax = Math.max(maxPassRate > 0 ? maxPassRate * 1.2 : 0, 1);
  const efficiencyAxisMax = niceUpperBound(Math.max(...progressChartPoints.map((point) => point.generation_efficiency), 0), 5);
  const bestScoreTrendMax = Math.max(...progressPoints.map((point) => Number(point.best_score) || 0), 0);
  const bestIcTrendMax = Math.max(...progressPoints.map((point) => Number(point.best_ic) || 0), 0);
  const bestScoreAxisMax = bestScoreTrendMax > 0 ? ceilToOneDecimal(bestScoreTrendMax * 1.2) : 1;
  const bestIcAxisMax = bestIcTrendMax > 0 ? ceilToOneDecimal(bestIcTrendMax * 1.5) : 1;
  const latestModelLab = modelLab?.latest ?? null;
  const lowCorrModelLab = modelLab?.low_corr_exploration ?? null;
  const visibleModelEntries = sortModelEntries(Object.entries(latestModelLab?.models || {}));
  const lowCorrModelEntries = sortModelEntries(Object.entries(lowCorrModelLab?.models || {}));
  const chartModelEntries = sortModelEntriesByGroup(Object.entries(latestModelLab?.models || {}));
  const lowCorrChartModelEntries = sortModelEntriesByGroup(Object.entries(lowCorrModelLab?.models || {}));
  const lowCorrBestPayload = lowCorrModelLab?.best_model ? lowCorrModelLab.models?.[lowCorrModelLab.best_model] : undefined;
  const fullBestPayload = latestModelLab?.best_model ? latestModelLab.models?.[latestModelLab.best_model] : undefined;
  const bestComboSummary = [
    latestModelLab?.best_model && fullBestPayload
      ? { lab: '全因子', model: latestModelLab.best_model, payload: fullBestPayload, selectedCount: latestModelLab.selected_factor_count }
      : null,
    lowCorrModelLab?.best_model && lowCorrBestPayload
      ? { lab: '低相关 8 因子', model: lowCorrModelLab.best_model, payload: lowCorrBestPayload, selectedCount: lowCorrModelLab.selected_factor_count }
      : null,
  ]
    .filter((item): item is { lab: string; model: string; payload: ModelLabModelSummary; selectedCount: number } => Boolean(item))
    .sort((a, b) => (b.payload.submit_Score ?? 0) - (a.payload.submit_Score ?? 0))[0];
  const bestComboLongShortPnlSeries = buildPnlDrawdownSeries(
    bestComboSummary?.payload.cumulative_curve,
    bestComboSummary?.payload.drawdown_curve,
  );
  const bestComboLongOnlyPnlSeries = buildPnlDrawdownSeries(
    bestComboSummary?.payload.long_only_cumulative_curve,
    bestComboSummary?.payload.long_only_drawdown_curve,
  );
  const modelComparison = chartModelEntries.map(([modelName, payload]) => ({
    model: modelName,
    modelLabel: stripComboName(modelName),
    group: getMethodGroup(modelName),
    score: payload.submit_Score ?? 0,
    ic: payload.submit_IC ?? 0,
    ir: payload.submit_IR ?? 0,
    tvr: getComboDisplayTvr(payload),
  }));
  const lowCorrModelComparison = lowCorrChartModelEntries.map(([modelName, payload]) => ({
    model: modelName,
    modelLabel: stripComboName(modelName),
    group: getMethodGroup(modelName),
    score: payload.submit_Score ?? 0,
    ic: payload.submit_IC ?? 0,
    ir: payload.submit_IR ?? 0,
    tvr: getComboDisplayTvr(payload),
  }));
  const modelComparisonSeparators = groupSeparators(modelComparison);
  const lowCorrModelComparisonSeparators = groupSeparators(lowCorrModelComparison);
  const trainValTestRows = chartModelEntries
    .map(([modelName, payload]) => {
      const trainScore = getPhaseMetric(payload, 'train', 'Score');
      const valScore = getPhaseMetric(payload, 'val', 'Score');
      const testScore = getPhaseMetric(payload, 'oos', 'Score');
      const trainIC = getPhaseMetric(payload, 'train', 'IC');
      const valIC = getPhaseMetric(payload, 'val', 'IC');
      const testIC = getPhaseMetric(payload, 'oos', 'IC');
      return {
        model: modelName,
        modelLabel: stripComboName(modelName),
        group: getMethodGroup(modelName),
        trainScore,
        valScore,
        testScore,
        trainIC,
        valIC,
        testIC,
        scoreRetention: trainScore > 0 ? (testScore / trainScore) * 100 : 0,
        icRetention: trainIC > 0 ? (testIC / trainIC) * 100 : 0,
      };
    })
    .filter((row) => row.trainScore || row.valScore || row.testScore || row.trainIC || row.valIC || row.testIC);
  const trainValTestSeparators = groupSeparators(trainValTestRows);
  const overfitScoreDomain: [number, number] = [
    0,
    Math.ceil(niceUpperBound(Math.max(...trainValTestRows.flatMap((row) => [row.trainScore, row.valScore, row.testScore]), 0), 1, 1.08)),
  ];
  const overfitIcDomain: [number, number] = [
    0,
    Math.ceil(niceUpperBound(Math.max(...trainValTestRows.flatMap((row) => [row.trainIC, row.valIC, row.testIC]), 0), 1, 1.12)),
  ];
  const scoreRetentionMedian = (() => {
    const values = trainValTestRows.map((row) => row.scoreRetention).filter((value) => Number.isFinite(value) && value > 0).sort((a, b) => a - b);
    return values.length ? values[Math.floor(values.length / 2)] : 0;
  })();
  const icRetentionMedian = (() => {
    const values = trainValTestRows.map((row) => row.icRetention).filter((value) => Number.isFinite(value) && value > 0).sort((a, b) => a - b);
    return values.length ? values[Math.floor(values.length / 2)] : 0;
  })();
  void overfitScoreDomain;
  void overfitIcDomain;
  void scoreRetentionMedian;
  void icRetentionMedian;
  const modelComparisonDomains = buildZeroAlignedDomains(
    modelComparison.map((row) => row.score),
    modelComparison.flatMap((row) => [row.ic, row.ir])
  );
  const lowCorrModelComparisonDomains = buildZeroAlignedDomains(
    lowCorrModelComparison.map((row) => row.score),
    lowCorrModelComparison.flatMap((row) => [row.ic, row.ir])
  );
  const methodTrendComparison = chartModelEntries.map(([modelName, payload]) => {
    const lowPayload = lowCorrModelLab?.models?.[modelName];
    return {
      model: modelName,
      modelLabel: stripComboName(modelName),
      group: getMethodGroup(modelName),
      fullScore: payload.submit_Score ?? 0,
      lowScore: lowPayload?.submit_Score ?? 0,
      fullIC: payload.submit_IC ?? 0,
      lowIC: lowPayload?.submit_IC ?? 0,
      fullIR: payload.submit_IR ?? 0,
      lowIR: lowPayload?.submit_IR ?? 0,
      fullTVR: getComboDisplayTvr(payload),
      lowTVR: getComboDisplayTvr(lowPayload),
    };
  });
  const methodTrendSeparators = groupSeparators(methodTrendComparison);
  const lowCorrFactorList = (lowCorrModelLab?.selected_factors || []).map((item) => item.run_id);
  const fullBestLongShortPnlSeries = buildPnlDrawdownSeries(fullBestPayload?.cumulative_curve, fullBestPayload?.drawdown_curve);
  const fullBestLongOnlyPnlSeries = buildPnlDrawdownSeries(fullBestPayload?.long_only_cumulative_curve, fullBestPayload?.long_only_drawdown_curve);
  const lowCorrBestLongShortPnlSeries = buildPnlDrawdownSeries(lowCorrBestPayload?.cumulative_curve, lowCorrBestPayload?.drawdown_curve);
  const lowCorrBestLongOnlyPnlSeries = buildPnlDrawdownSeries(lowCorrBestPayload?.long_only_cumulative_curve, lowCorrBestPayload?.long_only_drawdown_curve);
  const fusionLab = latestModelLab?.fusion_lab;
  const fusionResults = (fusionLab?.fusion_results || []).map((row) => ({
    model: row.model,
    modelLabel: stripComboName(row.model).replace(/^Fusion/, ''),
    mechanism: row.mechanism,
    score: row.Score ?? 0,
    ic: row.IC ?? 0,
    ir: row.IR ?? 0,
    tvr: row.TVR ?? 0,
    selectionObjective: row.selection_objective ?? 0,
    isValSelected: row.model === fusionLab?.selected_model,
    isOosBest: row.model === fusionLab?.best_oos_fusion_model,
  }));
  const fusionWeightRows = Object.entries(fusionLab?.fusion_weights || {})
    .map(([model, weight]) => ({ model, modelLabel: stripComboName(model), weight: Number(weight) || 0 }))
    .sort((a, b) => b.weight - a.weight);
  const fusionMatrixRows = fusionLab?.output_correlation_matrix || [];
  const fusionMatrixColumns = fusionMatrixRows[0]?.values?.map((item) => ({ model: item.model, label: item.label })) || [];
  const fusionCorrelationPairs = (fusionLab?.top_output_correlation_pairs || []).slice(0, 6);

  const statusData = Object.entries(knowledge?.status_breakdown ?? {}).map(([name, value]) => ({
    name,
    value,
  }));
  const scoreHistogram = buildHistogram(
    factors.filter((factor) => factor.status === 'ok').map((factor) => factor.Score),
    5,
    0
  );
  const icHistogram = buildHistogram(
    factors.filter((factor) => factor.status === 'ok').map((factor) => factor.IC),
    5,
    2
  );

  const handleStart = async () => {
    try {
      setPageMessage('');
      await fetchJson('/api/autoalpha/loop/start', {
        method: 'POST',
        body: JSON.stringify({ rounds, ideas, days, target_valid: targetValid }),
      });
      const nextStatus = await fetchJson<LoopStatus>('/api/autoalpha/loop/status');
      setStatus(nextStatus);
      setPageMessage('全量数据因子挖掘已启动。');
    } catch (error: any) {
      alert(error.message || '启动失败');
    }
  };

  const handleStop = async () => {
    try {
      await fetchJson('/api/autoalpha/loop/stop', { method: 'POST' });
      const nextStatus = await fetchJson<LoopStatus>('/api/autoalpha/loop/status');
      setStatus(nextStatus);
      setPageMessage('当前挖掘循环已停止。');
    } catch (error: any) {
      alert(error.message || '停止失败');
    }
  };

  const handleAddInspiration = async () => {
    if (!promptInput.trim()) {
      alert('请输入 Prompt、链接或研究启发。');
      return;
    }
    try {
      setPromptBusy(true);
      const data = await fetchJson<InspirationPayload>('/api/autoalpha/inspirations', {
        method: 'POST',
        body: JSON.stringify({ title: promptTitle, input: promptInput }),
      });
      setInspirations(data);
      setPromptTitle('');
      setPromptInput('');
      setPageMessage('灵感已写入 AutoAlpha Ideas，并会自动进入后续挖掘上下文。');
    } catch (error: any) {
      alert(error.message || '写入灵感失败');
    } finally {
      setPromptBusy(false);
    }
  };

  const handleSyncInspirations = async () => {
    try {
      setPromptBusy(true);
      const data = await fetchJson<InspirationPayload>('/api/autoalpha/inspirations/sync', {
        method: 'POST',
      });
      setInspirations(data);
      setPageMessage('AutoAlpha 目录下的 Prompt / 链接文件已同步。');
    } catch (error: any) {
      alert(error.message || '同步失败');
    } finally {
      setPromptBusy(false);
    }
  };

  const handleGenerateManualFactor = async () => {
    if (!promptInput.trim()) {
      alert('请输入想法或 DSL 公式。');
      return;
    }
    try {
      setManualBusy(true);
      const result = await fetchJson<{ factor: ManualFactor }>('/api/formula/execute', {
        method: 'POST',
        body: JSON.stringify({ input: promptInput }),
      });
      setManualFactor(result.factor);
      setPageMessage('单因子验证完成，结果已落入因子库。');
    } catch (error: any) {
      alert(error.message || '单因子验证失败');
    } finally {
      setManualBusy(false);
    }
  };

  return (
    <PageErrorBoundary>
      <div className="min-w-0 max-w-full space-y-6 overflow-x-hidden pb-10">
      <Panel
        title="AutoAlpha Research Cockpit"
        subtitle="主控制台与独立回测已收口到这里。现在统一在 AutoAlpha 页里完成灵感输入、全量挖掘、云端一致口径评估、rolling 模型实验与产出查看。"
        right={(
          <div className="rounded-full border border-border/60 bg-white/80 px-4 py-2 text-sm">
            <span className="text-muted-foreground">状态:</span>{' '}
            <span className={isRunning ? 'font-semibold text-emerald-600' : 'font-semibold text-slate-500'}>
              {isRunning ? '运行中' : '待机'}
            </span>
          </div>
        )}
        className="overflow-hidden bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.18),transparent_32%),radial-gradient(circle_at_top_right,rgba(16,185,129,0.16),transparent_30%),linear-gradient(180deg,rgba(255,255,255,0.95),rgba(248,250,252,0.92))]"
      >
        <div className="grid min-w-0 gap-4 md:grid-cols-2 xl:grid-cols-4">
          <StatCard label="已测试因子" value={String(status?.total_tested ?? 0)} helper="累计知识库记录" />
          <StatCard label="通过 Gate" value={String(status?.total_passing ?? 0)} helper={`通过率 ${formatPercent(knowledge?.pass_rate ?? 0)}`} accent="bg-emerald-50" />
          <StatCard label="最佳 Score" value={formatNumber(status?.best_score ?? 0, 2)} helper="按云端一致口径显示" accent="bg-sky-50" />
          <StatCard label="Ideas" value={String(inspirations?.count ?? 0)} helper="Manual / Paper / LLM" accent="bg-violet-50" />
        </div>
      </Panel>

      {pageMessage ? (
        <div className="rounded-3xl border border-sky-200 bg-sky-50 px-5 py-4 text-sm text-sky-700">{pageMessage}</div>
      ) : null}

      {balance?.warnings?.length ? (
        <div className="rounded-3xl border border-amber-200 bg-amber-50 px-5 py-4 text-sm text-amber-700">
          <div className="font-medium">额度接口提示</div>
          <div className="mt-2 space-y-1">
            {balance.warnings.map((warning) => (
              <div key={warning}>{warning}</div>
            ))}
          </div>
        </div>
      ) : null}

	      <div className="grid min-w-0 gap-6 xl:grid-cols-2 xl:items-stretch">
	        <div className="flex min-w-0 flex-col gap-6">
	        <Panel
	          title="额度包与成本"
	          subtitle="本页按第三方订阅接口返回的原始额度口径展示；换算关系为 $2 额度约对应 ¥1 实际中转消费。"
	          right={<div className={`rounded-full bg-emerald-500/10 px-4 py-2 text-sm font-medium ${quotaTone(balance?.quota_status ?? 'healthy')}`}>{balance?.quota_status ?? 'healthy'}</div>}
	        >
          {(() => {
            const pct = Math.min(balance?.used_pct ?? 0, 100);
            const barColor =
              (balance?.quota_status ?? 'healthy') === 'healthy' ? 'bg-emerald-500' :
              (balance?.quota_status) === 'warning' ? 'bg-amber-400' :
              (balance?.quota_status) === 'critical' ? 'bg-orange-500' : 'bg-red-500';
            return (
              <div className="mb-4">
                <div className="mb-1 flex items-center justify-between text-xs text-slate-500">
                  <span>额度占用</span>
                  <span className="tabular-nums font-medium">{formatPercent(pct)} 已使用</span>
                </div>
                <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${barColor}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })()}
          <div className="grid min-w-0 gap-4 md:grid-cols-3">
            <StatCard label="总额度" value={formatMoney(balance?.total_quota ?? 0)} helper="真实额度包口径" valueClassName="whitespace-nowrap text-[clamp(1.35rem,1.9vw,2.1rem)] tabular-nums" />
            <StatCard label="已用额度" value={formatMoney(balance?.used ?? 0)} helper={`${formatPercent(balance?.used_pct ?? 0)} 已使用`} accent="bg-orange-50" valueClassName="whitespace-nowrap text-[clamp(1.35rem,1.9vw,2.1rem)] tabular-nums" />
            <StatCard label="剩余额度" value={formatMoney(balance?.remaining ?? 0)} helper={`${formatPercent(balance?.remaining_pct ?? 0)} 剩余`} accent="bg-emerald-50" valueClassName="whitespace-nowrap text-[clamp(1.35rem,1.9vw,2.1rem)] tabular-nums" />
            <StatCard label="因子成本" value={formatMoney(balance?.avg_cost_per_factor ?? 0)} helper={`${formatInteger(balance?.avg_tokens_per_factor ?? 0)} tokens / 因子`} accent="bg-sky-50" valueClassName="whitespace-nowrap text-[clamp(1.25rem,1.7vw,1.9rem)] tabular-nums" />
            <StatCard label="有效因子成本" value={formatMoney(balance?.avg_cost_per_valid_factor ?? 0)} helper={`${formatInteger(balance?.avg_tokens_per_valid_factor ?? 0)} tokens / 通过因子`} accent="bg-violet-50" valueClassName="whitespace-nowrap text-[clamp(1.25rem,1.7vw,1.9rem)] tabular-nums" />
            <StatCard label="Token 消耗" value={formatInteger(balance?.est_total_tokens ?? 0)} helper={`${formatInteger(balance?.total_factors ?? 0)} 次调用 / 通过率 ${formatPercent(balance?.pass_rate ?? 0)}`} accent="bg-slate-50" valueClassName="whitespace-nowrap text-[clamp(1.25rem,1.7vw,1.9rem)] tabular-nums" />
          </div>
        </Panel>

	        <Panel title="研究进程综述" className="flex flex-1 flex-col">
          <div className="grid min-w-0 gap-5">
            <div className="grid min-w-0 gap-5 xl:grid-cols-2">
              <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
                <div className="mb-3 text-sm font-medium text-foreground">测试进度与通过数</div>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={progressChartPoints}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="timestamp" tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={formatAxisTime} minTickGap={28} />
                      <YAxis yAxisId="tested" tick={{ fill: '#64748b', fontSize: 11 }} domain={[0, 'dataMax + 5']} />
                      <YAxis
                        yAxisId="passing"
                        orientation="right"
                        domain={[0, passingAxisMax]}
                        tick={{ fill: COLORS.passing, fontSize: 11 }}
                      />
                      <Tooltip formatter={(value: number) => formatNumber(Number(value), 0)} />
                      <Legend />
                      <Area yAxisId="tested" type="monotone" dataKey="tested" name="tested" stroke={COLORS.tested} fill={COLORS.tested} fillOpacity={0.15} />
                      <Line
                        yAxisId="passing"
                        type="monotone"
                        dataKey="passing"
                        name="passing"
                        stroke={COLORS.passing}
                        strokeWidth={3}
                        dot={false}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
                <div className="mb-3 text-sm font-medium text-foreground">通过比例与生成效率</div>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={progressChartPoints} margin={{ top: 12, right: 18, bottom: 8, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="timestamp" tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={formatAxisTime} minTickGap={28} />
                      <YAxis yAxisId="rate" domain={[0, passRateAxisMax]} tick={{ fill: COLORS.passRate, fontSize: 11 }} tickFormatter={(value) => `${Number(value).toFixed(0)}%`} />
                      <YAxis yAxisId="efficiency" orientation="right" domain={[0, efficiencyAxisMax]} tick={{ fill: COLORS.efficiency, fontSize: 11 }} />
                      <Tooltip formatter={(value: number, name: string) => (name === '通过比例' ? `${formatNumber(Number(value), 2)}%` : `${formatNumber(Number(value), 2)} / 小时`)} />
                      <Legend />
                      <Line yAxisId="rate" type="monotone" dataKey="pass_rate" name="通过比例" stroke={COLORS.passRate} strokeWidth={3} dot={false} />
                      <Line yAxisId="efficiency" type="monotone" dataKey="generation_efficiency" name="平均每小时有效因子" stroke={COLORS.efficiency} strokeWidth={3} dot={false} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
              <div className="mb-3 text-sm font-medium text-foreground">最佳 Score / IC 趋势</div>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={progressPoints} margin={{ top: 8, right: 12, bottom: 8, left: 0 }}>
	                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
	                    <XAxis dataKey="timestamp" tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={formatAxisTime} minTickGap={28} />
	                    <YAxis yAxisId="score" domain={[0, bestScoreAxisMax]} tick={{ fill: COLORS.bestScore, fontSize: 11 }} tickFormatter={formatAxisOneDecimal} />
	                    <YAxis yAxisId="ic" orientation="right" domain={[0, bestIcAxisMax]} tick={{ fill: COLORS.ic, fontSize: 11 }} tickFormatter={formatAxisOneDecimal} />
	                    <Tooltip formatter={(value: number, name: string) => [formatNumber(Number(value), 1), name]} />
	                    <Legend />
	                    <Line yAxisId="score" type="monotone" dataKey="best_score" name="最佳 Score" stroke={COLORS.bestScore} strokeWidth={3} dot={false} />
	                    <Line yAxisId="ic" type="monotone" dataKey="best_ic" name="最佳 IC" stroke={COLORS.ic} strokeWidth={2.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="grid min-w-0 gap-5 lg:grid-cols-3">
              <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
                <div className="mb-3 text-sm font-medium text-foreground">Score 分布</div>
                <div className="h-44">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={scoreHistogram}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="tickLabel" tick={{ fill: '#64748b', fontSize: 10 }} interval={0} height={28} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
                      <Tooltip labelFormatter={(_, payload) => payload?.[0]?.payload?.label || ''} />
                      <Bar dataKey="count" fill={COLORS.score} radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
                <div className="mb-3 text-sm font-medium text-foreground">IC 分布</div>
                <div className="h-44">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={icHistogram}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="tickLabel" tick={{ fill: '#64748b', fontSize: 10 }} interval={0} height={28} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
                      <Tooltip labelFormatter={(_, payload) => payload?.[0]?.payload?.label || ''} />
                      <Bar dataKey="count" fill={COLORS.ic} radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
                <div className="mb-3 text-sm font-medium text-foreground">状态结构</div>
                <div className="h-44">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={statusData} dataKey="value" nameKey="name" innerRadius={42} outerRadius={74} paddingAngle={3}>
                        {statusData.map((item, index) => (
                          <Cell key={item.name} fill={[COLORS.passing, COLORS.tested, COLORS.error, COLORS.histogram][index % 4]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
	                  </ResponsiveContainer>
	                </div>
	              </div>
	            </div>

	            <div className="min-w-0 rounded-3xl border border-emerald-100 bg-emerald-50/70 p-4">
	              <div className="flex flex-wrap items-start justify-between gap-3">
	                <div className="min-w-0">
	                  <div className="text-[11px] uppercase tracking-[0.18em] text-emerald-700">当前最佳 Combo</div>
	                  <div className="mt-1 truncate text-sm font-semibold text-foreground">
	                    {bestComboSummary ? `${bestComboSummary.lab} · ${stripComboName(bestComboSummary.model)}` : '等待 Model Lab 结果'}
	                  </div>
	                </div>
	                {bestComboSummary ? (
	                  <button
	                    onClick={() => setComboModal({ modelName: bestComboSummary.model, lab: bestComboSummary.lab === '低相关 8 因子' ? 'low' : 'full' })}
	                    className="rounded-full bg-white px-3 py-1 text-xs font-medium text-emerald-700 shadow-sm transition-colors hover:bg-emerald-100"
	                  >
	                    查看详情
	                  </button>
	                ) : null}
	              </div>
		              {bestComboSummary ? (
		                <>
		                  <div className="mt-4 grid gap-3 text-sm text-slate-700 sm:grid-cols-5">
		                    <div className="rounded-2xl bg-white/80 p-3">
		                      <div className="text-[11px] text-muted-foreground">Score</div>
		                      <div className="mt-1 font-semibold text-emerald-700">{formatNumber(bestComboSummary.payload.submit_Score ?? 0, 1)}</div>
		                    </div>
		                    <div className="rounded-2xl bg-white/80 p-3">
		                      <div className="text-[11px] text-muted-foreground">IC</div>
		                      <div className="mt-1 font-semibold text-sky-700">{formatNumber(bestComboSummary.payload.submit_IC ?? 0, 1)}</div>
		                    </div>
		                    <div className="rounded-2xl bg-white/80 p-3">
		                      <div className="text-[11px] text-muted-foreground">IR</div>
		                      <div className="mt-1 font-semibold text-slate-800">{formatNumber(bestComboSummary.payload.submit_IR ?? 0, 1)}</div>
		                    </div>
		                    <div className="rounded-2xl bg-white/80 p-3">
		                      <div className="text-[11px] text-muted-foreground">TVR</div>
		                      <div className="mt-1 font-semibold text-slate-800">{formatNumber(getComboDisplayTvr(bestComboSummary.payload), 1)}</div>
		                    </div>
		                    <div className="rounded-2xl bg-white/80 p-3">
		                      <div className="text-[11px] text-muted-foreground">因子数</div>
		                      <div className="mt-1 font-semibold text-slate-800">{bestComboSummary.selectedCount}</div>
		                    </div>
		                  </div>
		                  <div className="mt-4 space-y-3 rounded-2xl bg-white/80 p-3">
		                    <PnlDrawdownChart title="2024 OOS 多空 PnL + Max DD" data={bestComboLongShortPnlSeries} pnlColor="#2563eb" heightClass="h-[170px]" />
		                    <PnlDrawdownChart title="2024 OOS 纯多头 PnL + Max DD" data={bestComboLongOnlyPnlSeries} pnlColor="#0f766e" heightClass="h-[170px]" />
		                  </div>
		                </>
		              ) : (
		                <div className="mt-3 text-sm text-muted-foreground">Model Lab 产出后会自动展示全因子或低相关部分因子中 Score 最高的 2024 OOS combo。</div>
		              )}
	            </div>
	          </div>
	        </Panel>

        </div>

	        <div className="flex min-w-0 flex-col gap-6">
	        <Panel
          title="Prompt Lab"
          subtitle="现在可以直接在 AutoAlpha 页面输入文字链接或纯提示词。内容会落到 AutoAlpha Ideas，并自动进入后续因子挖掘上下文。"
          right={(
            <div className="rounded-full bg-slate-100 px-4 py-2 text-sm text-slate-600">
              <BrainCircuit className="mr-2 inline h-4 w-4" />
              {inspirations?.count ?? 0} 条灵感
            </div>
          )}
        >
          <div className="grid min-w-0 items-start gap-5 lg:grid-cols-[minmax(0,1.35fr)_minmax(260px,0.65fr)]">
            <div className="min-w-0 space-y-4">
              <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
                <div className="grid gap-4">
                  <label>
                    <div className="text-xs text-muted-foreground">标题（可选）</div>
                    <input
                      value={promptTitle}
                      onChange={(event) => setPromptTitle(event.target.value)}
                      placeholder="例如：盘中流动性枯竭 / 某篇文章链接"
                      className="mt-2 w-full rounded-2xl border border-border/60 bg-slate-50 px-4 py-3 text-sm outline-none"
                    />
                  </label>
                  <label>
                    <div className="text-xs text-muted-foreground">Prompt / 链接 / DSL</div>
                    <textarea
                      value={promptInput}
                      onChange={(event) => setPromptInput(event.target.value)}
                      placeholder="输入研究灵感、文章链接、市场观察，或直接输入一段 DSL。保存后会进入 AutoAlpha Ideas。"
                      className="mt-2 h-40 w-full rounded-2xl border border-border/60 bg-slate-50 px-4 py-3 text-sm outline-none"
                    />
                  </label>
                </div>
                <div className="mt-4 flex flex-wrap gap-3">
                  <button onClick={handleAddInspiration} disabled={promptBusy} className="rounded-2xl bg-slate-950 px-4 py-3 text-sm text-white transition-colors hover:bg-slate-800 disabled:opacity-50">
                    {promptBusy ? '处理中...' : '加入 Ideas'}
                  </button>
                  <button onClick={handleGenerateManualFactor} disabled={manualBusy} className="rounded-2xl border border-border/60 bg-white px-4 py-3 text-sm text-foreground transition-colors hover:bg-slate-50 disabled:opacity-50">
                    <Wand2 className="mr-2 inline h-4 w-4" />
                    {manualBusy ? '验证中...' : '立即验证单因子'}
                  </button>
                  <button onClick={handleSyncInspirations} disabled={promptBusy} className="rounded-2xl border border-border/60 bg-white px-4 py-3 text-sm text-foreground transition-colors hover:bg-slate-50 disabled:opacity-50">
                    <FolderSync className="mr-2 inline h-4 w-4" />
                    同步目录文件
                  </button>
                </div>
              </div>

              <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
                <div className="mb-3 flex items-center justify-between gap-2">
                  <div className="text-sm font-medium text-foreground">目录与数据库</div>
                  <span className="text-[11px] text-muted-foreground">上下文预览</span>
                </div>
                <div className="grid gap-2 text-sm">
                  <div className="grid gap-2 sm:grid-cols-2">
                    <div className="min-w-0 rounded-2xl bg-slate-50 p-3">
                      <div className="text-[11px] text-muted-foreground">Prompt Dir</div>
                      <div className="mt-1 overflow-x-auto whitespace-nowrap font-mono text-[11px] leading-5 text-foreground">{inspirations?.prompt_dir || '--'}</div>
                    </div>
                    <div className="min-w-0 rounded-2xl bg-slate-50 p-3">
                      <div className="text-[11px] text-muted-foreground">SQLite DB</div>
                      <div className="mt-1 overflow-x-auto whitespace-nowrap font-mono text-[11px] leading-5 text-foreground">{inspirations?.database_path || '--'}</div>
                    </div>
                  </div>
                  <div className="rounded-2xl bg-slate-50 p-3">
                    <div className="text-[11px] text-muted-foreground">注入 LLM 的上下文预览</div>
                    <div className="mt-2 max-h-32 overflow-y-auto whitespace-pre-wrap break-words pr-2 text-[11px] leading-5 text-foreground">
                      {inspirations?.prompt_context_preview || '当前没有灵感上下文。'}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="min-w-0 space-y-4">
              <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                    <Sparkles className="h-4 w-4 text-violet-500" />
                    最近灵感
                  </div>
                  <span className="text-[11px] text-muted-foreground">{inspirations?.count ?? 0} 条</span>
                </div>
                <div className="max-h-72 overflow-y-auto space-y-1.5 pr-0.5">
                  {(inspirations?.items || []).length === 0 ? (
                    <div className="text-xs text-muted-foreground py-1">还没有灵感记录。</div>
                  ) : (
                    inspirations?.items.map((item) => (
                      <div key={item.id} className="rounded-xl border border-border/40 bg-slate-50/80 px-2.5 py-2">
                        <div className="flex items-baseline justify-between gap-2">
                          {item.source_type === 'paper' && item.source?.startsWith('http') ? (
                            <a
                              href={item.source}
                              target="_blank"
                              rel="noreferrer"
                              className="truncate text-xs font-medium leading-tight text-sky-700 underline-offset-2 hover:underline"
                              title={item.source}
                            >
                              {item.title}
                            </a>
                          ) : (
                            <div className="truncate text-xs font-medium text-foreground leading-tight">{item.title}</div>
                          )}
                          <div className="shrink-0 text-[10px] text-muted-foreground">{formatDateTime(item.created_at)}</div>
                        </div>
                        <div className="mt-0.5 line-clamp-2 text-[11px] leading-relaxed text-slate-600">
                          {item.summary || truncate(item.content, 80)}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div className="min-h-[132px] rounded-3xl border border-border/50 bg-white/90 p-4">
                <div className="mb-3 flex items-center gap-2 text-sm font-medium text-foreground">
                  <FlaskConical className="h-4 w-4 text-sky-500" />
                  单因子即时验证
                </div>
                {!manualFactor ? (
                  <div className="text-sm text-muted-foreground">输入 Prompt 或 DSL 后，可以在这里直接看到单因子的云端一致口径结果。</div>
                ) : (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-3">
                      <div className="rounded-2xl bg-slate-50 p-3">
                        <div className="text-[11px] text-muted-foreground">Score</div>
                        <div className="mt-2 text-xl font-semibold text-foreground">{formatNumber(manualFactor.Score ?? 0, 2)}</div>
                      </div>
                      <div className="rounded-2xl bg-slate-50 p-3">
                        <div className="text-[11px] text-muted-foreground">Cloud TVR</div>
                        <div className="mt-2 text-xl font-semibold text-foreground">{formatNumber(manualFactor.Turnover ?? 0, 1)}</div>
                      </div>
                      <div className="rounded-2xl bg-slate-50 p-3">
                        <div className="text-[11px] text-muted-foreground">IC / IR</div>
                        <div className="mt-2 text-sm font-semibold text-foreground">{formatNumber(manualFactor.IC ?? 0, 3)} / {formatNumber(manualFactor.IR ?? 0, 2)}</div>
                      </div>
                      <div className={`rounded-2xl p-3 ${manualFactor.submission_ready_flag ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'}`}>
                        <div className="text-[11px]">提交状态</div>
                        <div className="mt-2 text-sm font-semibold">{manualFactor.submission_ready_flag ? '可直接提交' : '研究候选'}</div>
                      </div>
                    </div>
                    <div className="rounded-2xl bg-slate-50 p-3">
                      <div className="text-[11px] text-muted-foreground">公式</div>
                      <div className="mt-2 break-all font-mono text-xs leading-6 text-foreground">{manualFactor.formula || '--'}</div>
                    </div>
                    {manualFactor.submission_path ? (
                      <div className="rounded-2xl bg-slate-50 p-3">
                        <div className="text-[11px] text-muted-foreground">提交文件</div>
                        <div className="mt-2 break-all text-xs text-foreground">{manualFactor.submission_path}</div>
                      </div>
                    ) : null}
                  </div>
                )}
              </div>
            </div>
          </div>
        </Panel>

	        <Panel title="循环控制与实时日志" subtitle="直接启动全量数据挖掘；rounds=0 表示持续运行，直到达到目标有效因子数或手动停止。" className="flex flex-1 flex-col">
          <div className="grid min-w-0 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <label className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4 text-sm">
              <div className="text-xs text-muted-foreground">轮数 rounds</div>
              <input type="number" min={0} max={2000} value={rounds} onChange={(event) => setRounds(Math.max(0, Number(event.target.value)))} disabled={isRunning} className="mt-3 w-full rounded-2xl border border-border/60 bg-slate-50 px-3 py-2 text-base outline-none disabled:opacity-50" />
              <div className="mt-2 text-[11px] text-muted-foreground">0 表示长跑模式</div>
            </label>
            <label className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4 text-sm">
              <div className="text-xs text-muted-foreground">每轮 ideas</div>
              <input type="number" min={1} max={20} value={ideas} onChange={(event) => setIdeas(Number(event.target.value))} disabled={isRunning} className="mt-3 w-full rounded-2xl border border-border/60 bg-slate-50 px-3 py-2 text-base outline-none disabled:opacity-50" />
            </label>
            <label className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4 text-sm">
              <div className="text-xs text-muted-foreground">交易日 days</div>
              <input type="number" min={0} value={days} onChange={(event) => setDays(Math.max(0, Number(event.target.value)))} disabled={isRunning} className="mt-3 w-full rounded-2xl border border-border/60 bg-slate-50 px-3 py-2 text-base outline-none disabled:opacity-50" />
              <div className="mt-2 text-[11px] text-muted-foreground">输入 0 即使用全量数据</div>
            </label>
            <label className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4 text-sm">
              <div className="text-xs text-muted-foreground">目标有效因子</div>
              <input
                type="number"
                min={0}
                max={1000}
                value={targetValid}
                onChange={(event) => {
                  setTargetValid(Math.max(0, Number(event.target.value)));
                  setTargetValidSource('manual');
                }}
                disabled={isRunning}
                className="mt-3 w-full rounded-2xl border border-border/60 bg-slate-50 px-3 py-2 text-base outline-none disabled:opacity-50"
              />
              <div className="mt-2 text-[11px] text-muted-foreground">
                当前自动建议 {suggestedTargetValid || 0}；手动填 0 表示不设目标
              </div>
            </label>
          </div>

          <div className="mt-4 grid min-w-0 gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(132px,180px)] xl:grid-cols-[minmax(0,1fr)_180px]">
            <button onClick={handleStart} disabled={isRunning} className="rounded-3xl bg-slate-950 px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50">
              {isRunning ? '挖掘中...' : '启动全量挖掘'}
            </button>
            <button onClick={handleStop} disabled={!isRunning} className="rounded-3xl border border-red-300 bg-red-50 px-4 py-3 text-sm font-medium text-red-600 transition-colors hover:bg-red-100 disabled:cursor-not-allowed disabled:opacity-50">
              停止循环
            </button>
          </div>

          <div className="mt-4 grid min-w-0 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,0.85fr)]">
            <StatCard label="最近更新" value={formatDateTime(status?.updated_at)} helper={status?.pid ? `PID ${status.pid}` : '轻量状态轮询间隔 6 秒'} valueClassName="text-xl" />
            <StatCard label="云端一致说明" value="TVR / Score 已对齐" helper="submission-like 提交评分链路" accent="bg-emerald-50" valueClassName="text-[clamp(1.2rem,2vw,1.75rem)] leading-tight" />
          </div>

          <div className="mt-4 min-w-0 rounded-3xl border border-border/50 bg-white/90 p-3 sm:p-4">
            <div className="mb-2 flex min-w-0 items-center justify-between gap-3">
              <div className="text-sm font-medium text-foreground">实时日志</div>
              <div className="shrink-0 text-xs text-muted-foreground">{status?.logs?.slice(-1)[0] ? '实时滚动' : '等待启动'}</div>
            </div>
            <LogPanel logs={status?.logs ?? []} />
          </div>
        </Panel>
        </div>
      </div>

        <Panel title="Exploratory OOS Combo Lab" subtitle="使用全部有效因子，固定以 2022-2023 为训练期、2024 为模拟 OOS 期，只保留真实 OOS 阶段也能实现的 combo 方法，并展示 2024 的 OOS Score / IC / IR / TVR 及相关性结构。">
          <div className="grid min-w-0 gap-5 xl:grid-cols-2">
            <div className="flex h-[700px] min-w-0 flex-col rounded-3xl border border-border/50 bg-white/90 p-4">
	              <div className="mb-3 flex items-center gap-2 text-sm font-medium text-foreground">
	                <Sparkles className="h-4 w-4 text-emerald-500" />
	                OOS Combo 摘要
	              </div>
	              <div className="mb-3 text-xs leading-5 text-muted-foreground">
	                使用目前的全量因子计算展示。
	              </div>
              <div className="grid min-w-0 gap-4 md:grid-cols-2">
                <StatCard label="Run ID" value={latestModelLab?.run_id || '--'} helper={latestModelLab ? formatDateTime(latestModelLab.created_at) : '还没有实验'} valueClassName="text-lg" />
                <StatCard label="因子总数" value={String(latestModelLab?.selected_factor_count ?? 0)} helper={latestModelLab ? `train ${latestModelLab.train_period_start} → ${latestModelLab.train_period_end}` : '使用全部有效因子'} accent="bg-violet-50" />
                <StatCard label="最佳 IC" value={formatNumber(latestModelLab?.best_ic ?? 0, 2)} helper={latestModelLab ? `eval ${latestModelLab.eval_period_start} → ${latestModelLab.eval_period_end}` : '2024 OOS'} accent="bg-sky-50" />
                <StatCard label="最佳 Score" value={formatNumber(latestModelLab?.best_score ?? 0, 2)} helper={latestModelLab?.best_model || '等待实验结果'} accent="bg-emerald-50" />
              </div>

	              <div className="mt-4 min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
	                {visibleModelEntries.map(([modelName, payload]) => {
	                  const isBestModel = modelName === latestModelLab?.best_model;
	                  const groupMeta = getMethodGroupMeta(modelName);
	                  return (
	                    <button
	                      key={modelName}
	                      onClick={() => setComboModal({ modelName, lab: 'full' })}
	                      className={`w-full rounded-2xl p-3 text-left transition-colors ${groupMeta.rowClass}`}
	                    >
	                      <div className="flex min-w-0 items-center justify-between gap-3">
		                        <div className="flex min-w-0 items-center gap-2">
		                          <div className="min-w-0 truncate font-medium text-foreground">{modelName}</div>
		                          {isBestModel ? <span className="shrink-0 rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-medium text-emerald-700">BEST</span> : null}
		                        </div>
	                        <div className="shrink-0 text-xs text-muted-foreground">TVR {formatNumber(getComboDisplayTvr(payload), 1)}</div>
                      </div>
                      <div className="mt-2 grid gap-2 text-[11px] text-slate-600 sm:grid-cols-2 xl:grid-cols-4">
                        <div>OOS Score {formatNumber(payload.submit_Score ?? 0, 2)}</div>
                        <div>OOS IC {formatNumber(payload.submit_IC ?? 0, 2)}</div>
                        <div>OOS IR {formatNumber(payload.submit_IR ?? 0, 2)}</div>
                        <div>Hit {formatPercent(payload.hit_ratio * 100)}</div>
                      </div>
                    </button>
                  );
                })}
                {!visibleModelEntries.length ? (
                  <div className="text-sm text-muted-foreground">当前还没有 OOS combo 实验结果。</div>
                ) : null}
              </div>
            </div>

	            <div className="flex h-[700px] min-w-0 flex-col rounded-3xl border border-border/50 bg-white/90 p-4">
	              <div className="mb-2 text-sm font-medium text-foreground">低相关组合探索</div>
	              <div className="mb-3 text-xs leading-5 text-muted-foreground">
	                固定使用 8 个低相关提交因子，导出 pq 文件会附加 `lowcorr8` 后缀，便于和全因子结果区分。
	              </div>
              <div className="grid min-w-0 gap-4 md:grid-cols-2">
                <StatCard label="探索 Run ID" value={lowCorrModelLab?.run_id || '--'} helper={lowCorrModelLab ? formatDateTime(lowCorrModelLab.created_at) : '等待实验'} valueClassName="text-lg" />
                <StatCard label="探索因子数" value={String(lowCorrModelLab?.selected_factor_count ?? 0)} helper="固定低相关 8 因子" accent="bg-violet-50" />
                <StatCard label="探索最佳 IC" value={formatNumber(lowCorrModelLab?.best_ic ?? 0, 2)} helper={lowCorrModelLab?.best_model || '--'} accent="bg-sky-50" />
                <StatCard label="探索最佳 Score" value={formatNumber(lowCorrModelLab?.best_score ?? 0, 2)} helper={lowCorrBestPayload ? `相对全因子 ${formatNumber((lowCorrBestPayload.submit_Score ?? 0) - (fullBestPayload?.submit_Score ?? 0), 2)}` : '--'} accent="bg-emerald-50" />
              </div>
              <div className="mt-4 max-h-24 overflow-y-auto rounded-2xl bg-slate-50 p-3 text-[11px] leading-6 text-slate-600">
                使用因子:
                <div className="mt-2 break-all font-mono text-[11px] text-slate-700">
                  {lowCorrFactorList.length ? lowCorrFactorList.join(' / ') : '--'}
                </div>
              </div>
	              <div className="mt-4 min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
	                {lowCorrModelEntries.map(([modelName, payload]) => {
	                  const isBestModel = modelName === lowCorrModelLab?.best_model;
	                  const groupMeta = getMethodGroupMeta(modelName);
	                  return (
	                    <button
	                      key={modelName}
	                      onClick={() => setComboModal({ modelName, lab: 'low' })}
	                      className={`w-full rounded-2xl p-3 text-left transition-colors ${groupMeta.rowClass}`}
	                    >
	                      <div className="flex min-w-0 items-center justify-between gap-3">
		                        <div className="flex min-w-0 items-center gap-2">
		                          <div className="min-w-0 truncate font-medium text-foreground">{modelName}</div>
		                          {isBestModel ? <span className="shrink-0 rounded-full bg-violet-100 px-2 py-0.5 text-[10px] font-medium text-violet-700">BEST</span> : null}
		                        </div>
	                        <div className="shrink-0 text-xs text-muted-foreground">TVR {formatNumber(getComboDisplayTvr(payload), 1)}</div>
                      </div>
                      <div className="mt-2 grid gap-2 text-[11px] text-slate-600 sm:grid-cols-2 xl:grid-cols-4">
                        <div>OOS Score {formatNumber(payload.submit_Score ?? 0, 2)}</div>
                        <div>OOS IC {formatNumber(payload.submit_IC ?? 0, 2)}</div>
                        <div>OOS IR {formatNumber(payload.submit_IR ?? 0, 2)}</div>
                        <div>Hit {formatPercent((payload.hit_ratio ?? 0) * 100)}</div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="mt-5 grid min-w-0 gap-5 xl:grid-cols-2">
            <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
              <div className="mb-3 text-sm font-medium text-foreground">全因子最佳 combo：2024 OOS PnL / Max DD</div>
              <div className="mb-3 text-xs leading-5 text-muted-foreground">
                上图为 top-bottom 多空组合累计 PnL，下图为只持有预测最高 top 20% 的纯多头累计 PnL；红线均为对应回撤曲线。
              </div>
              <div className="space-y-3">
                <PnlDrawdownChart title="多空组合 PnL + Max DD" data={fullBestLongShortPnlSeries} pnlColor="#2563eb" />
                <PnlDrawdownChart title="纯多头 PnL + Max DD" data={fullBestLongOnlyPnlSeries} pnlColor="#0f766e" />
              </div>
	              <div className="mt-4 grid gap-3 text-sm text-slate-700 sm:grid-cols-4">
	                <div className="rounded-2xl bg-slate-50 p-3">Score {formatNumber(fullBestPayload?.submit_Score ?? 0, 2)}</div>
	                <div className="rounded-2xl bg-slate-50 p-3">IC {formatNumber(fullBestPayload?.submit_IC ?? 0, 2)}</div>
	                <div className="rounded-2xl bg-slate-50 p-3">IR {formatNumber(fullBestPayload?.submit_IR ?? 0, 2)}</div>
	                <div className="rounded-2xl bg-slate-50 p-3">TVR {formatNumber(getComboDisplayTvr(fullBestPayload), 1)}</div>
	              </div>
            </div>

            <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
              <div className="mb-3 text-sm font-medium text-foreground">低相关 8 因子最佳 combo：2024 OOS PnL / Max DD</div>
              <div className="mb-3 text-xs leading-5 text-muted-foreground">
                同样固定 2022-2023 训练、2024 只检验；上图多空，下图纯多头，均用红线标出对应 Max DD 路径。
              </div>
              <div className="space-y-3">
                <PnlDrawdownChart title="多空组合 PnL + Max DD" data={lowCorrBestLongShortPnlSeries} pnlColor="#7c3aed" />
                <PnlDrawdownChart title="纯多头 PnL + Max DD" data={lowCorrBestLongOnlyPnlSeries} pnlColor="#0f766e" />
              </div>
	              <div className="mt-4 grid gap-3 text-sm text-slate-700 sm:grid-cols-5">
	                <div className="rounded-2xl bg-slate-50 p-3">TVR {formatNumber(getComboDisplayTvr(lowCorrBestPayload), 1)}</div>
	                <div className="rounded-2xl bg-slate-50 p-3">Score 变化 {formatNumber((lowCorrBestPayload?.submit_Score ?? 0) - (fullBestPayload?.submit_Score ?? 0), 2)}</div>
	                <div className="rounded-2xl bg-slate-50 p-3">IC 变化 {formatNumber((lowCorrBestPayload?.submit_IC ?? 0) - (fullBestPayload?.submit_IC ?? 0), 2)}</div>
	                <div className="rounded-2xl bg-slate-50 p-3">IR 变化 {formatNumber((lowCorrBestPayload?.submit_IR ?? 0) - (fullBestPayload?.submit_IR ?? 0), 2)}</div>
	                <div className="rounded-2xl bg-slate-50 p-3">TVR 变化 {formatNumber(getComboDisplayTvr(lowCorrBestPayload) - getComboDisplayTvr(fullBestPayload), 2)}</div>
	              </div>
	            </div>
	          </div>

		          <div className="mt-5 min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
		            <div className="mb-1 text-sm font-medium text-foreground">Model Fusion Lab：输出相关性与融合结果</div>
		            <div className="mb-3 text-xs leading-5 text-muted-foreground">
		              先计算冻结模型输出之间的相关度，再用 2022-2023 Val 指标和相关性惩罚生成融合权重；2024 标签只用于最终展示，不参与权重选择。
		            </div>
		            {fusionResults.length ? (
		              <div className="grid min-w-0 gap-5 xl:grid-cols-[1.15fr_0.85fr]">
		                <div className="min-w-0">
		                  <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-slate-600">
		                    <span className="rounded-full bg-emerald-50 px-2 py-1 text-emerald-700">Val 选择：{fusionLab?.selected_mechanism || '--'}</span>
		                    <span className="rounded-full bg-sky-50 px-2 py-1 text-sky-700">2024 诊断最佳：{fusionLab?.best_oos_fusion_mechanism || '--'}</span>
		                    <span className="rounded-full bg-slate-50 px-2 py-1 text-slate-600">当前最佳仍为 {latestModelLab?.best_model || '--'}</span>
		                  </div>
		                  <div className="h-[300px]">
		                    <ResponsiveContainer width="100%" height="100%">
		                      <BarChart data={fusionResults} margin={{ top: 8, right: 16, bottom: 52, left: 0 }}>
		                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
		                        <XAxis dataKey="modelLabel" tick={{ fill: '#64748b', fontSize: 11 }} angle={-16} textAnchor="end" height={86} interval={0} />
		                        <YAxis yAxisId="left" allowDecimals={false} tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={formatAxisInteger} />
		                        <YAxis yAxisId="right" orientation="right" tick={{ fill: '#64748b', fontSize: 11 }} />
		                        <Tooltip formatter={(value: number, name: string) => [formatNumber(Number(value), name === 'OOS Score' ? 1 : 2), name]} />
		                        <Legend />
		                        <Bar yAxisId="left" dataKey="score" name="OOS Score" fill="#0f766e" radius={[6, 6, 0, 0]}>
		                          {fusionResults.map((row) => (
		                            <Cell key={`fusion-score-${row.model}`} fill={row.isOosBest ? '#059669' : row.isValSelected ? '#2563eb' : '#94a3b8'} />
		                          ))}
		                        </Bar>
		                        <Bar yAxisId="right" dataKey="ic" name="OOS IC" fill="#2563eb" radius={[6, 6, 0, 0]} />
		                        <Bar yAxisId="right" dataKey="ir" name="OOS IR" fill="#f59e0b" radius={[6, 6, 0, 0]} />
		                      </BarChart>
		                    </ResponsiveContainer>
		                  </div>
		                  <div className="mt-3 grid gap-3 text-sm text-slate-700 sm:grid-cols-4">
		                    <div className="rounded-2xl bg-slate-50 p-3">最佳融合 Score {formatNumber(Math.max(...fusionResults.map((row) => row.score)), 2)}</div>
		                    <div className="rounded-2xl bg-slate-50 p-3">最佳融合 IC {formatNumber(Math.max(...fusionResults.map((row) => row.ic)), 2)}</div>
		                    <div className="rounded-2xl bg-slate-50 p-3">融合候选 {fusionResults.length}</div>
		                    <div className="rounded-2xl bg-slate-50 p-3">相关矩阵 {fusionMatrixRows.length}×{fusionMatrixColumns.length}</div>
		                  </div>
		                </div>
		                <div className="min-w-0">
		                  <div className="mb-2 text-xs font-medium text-slate-600">Val 选择机制权重</div>
		                  <div className="space-y-2">
		                    {fusionWeightRows.map((row) => (
		                      <div key={row.model}>
		                        <div className="mb-1 flex items-center justify-between gap-3 text-[11px] text-slate-600">
		                          <span className="truncate">{row.modelLabel}</span>
		                          <span>{formatNumber(row.weight * 100, 1)}%</span>
		                        </div>
		                        <div className="h-2 overflow-hidden rounded-full bg-slate-100">
		                          <div className="h-full rounded-full bg-emerald-500" style={{ width: `${Math.max(2, row.weight * 100)}%` }} />
		                        </div>
		                      </div>
		                    ))}
		                  </div>
		                  <div className="mt-4 text-xs leading-5 text-muted-foreground">{fusionLab?.leakage_note || ''}</div>
		                </div>
		              </div>
		            ) : (
		              <div className="rounded-2xl bg-slate-50 p-4 text-sm text-muted-foreground">暂无融合实验结果。</div>
		            )}
		            {fusionMatrixRows.length ? (
		              <div className="mt-5 grid min-w-0 gap-5 xl:grid-cols-[1.2fr_0.8fr]">
		                <div className="min-w-0 overflow-x-auto rounded-2xl border border-slate-100 bg-slate-50 p-3">
		                  <div className="mb-3 flex flex-wrap gap-2 text-[11px]">
		                    {(['linear', 'ml', 'combo'] as MethodGroupKey[]).map((group) => (
		                      <span key={`fusion-legend-${group}`} className={`${METHOD_GROUP_META[group].textClass} rounded-full bg-white px-2 py-1 font-medium`}>
		                        {METHOD_GROUP_META[group].label}
		                      </span>
		                    ))}
		                    <span className="rounded-full bg-white px-2 py-1 font-medium text-slate-500">色轴：|corr| 0 蓝 / 1 红</span>
		                  </div>
		                  <div
		                    className="grid gap-1 text-[10px]"
		                    style={{ gridTemplateColumns: `120px repeat(${fusionMatrixColumns.length}, minmax(48px, 1fr))` }}
		                  >
		                    <div />
		                    {fusionMatrixColumns.map((col) => (
		                      <div
		                        key={`fusion-col-${col.model}`}
		                        className={`truncate text-center font-semibold ${getMethodGroupMeta(col.model).textClass}`}
		                        title={`${col.model} · ${getMethodGroupMeta(col.model).label}`}
		                      >
		                        {col.label}
		                      </div>
		                    ))}
		                    {fusionMatrixRows.map((row) => (
		                      <React.Fragment key={`fusion-row-${row.model}`}>
		                        <div
		                          className={`truncate pr-2 text-right font-semibold ${getMethodGroupMeta(row.model).textClass}`}
		                          title={`${row.model} · ${getMethodGroupMeta(row.model).label}`}
		                        >
		                          {row.label}
		                        </div>
		                        {row.values.slice(0, fusionMatrixColumns.length).map((cell) => (
		                          <div
		                            key={`${row.model}-${cell.model}`}
		                            className="rounded-md px-1 py-2 text-center font-medium text-white"
		                            style={{ backgroundColor: corrCellColor(cell.corr) }}
		                            title={`${row.model} vs ${cell.model}: corr ${formatNumber(cell.corr, 3)}, |corr| ${formatNumber(Math.abs(cell.corr), 3)}`}
		                          >
		                            {formatNumber(Math.abs(cell.corr), 2)}
		                          </div>
		                        ))}
		                      </React.Fragment>
		                    ))}
		                  </div>
		                </div>
		                <div className="min-w-0 rounded-2xl bg-slate-50 p-3">
		                  <div className="mb-2 text-xs font-medium text-slate-600">最高相关输出对</div>
		                  <div className="space-y-2">
		                    {fusionCorrelationPairs.map((row) => (
		                      <div key={`${row.left}-${row.right}`} className="rounded-xl bg-white px-3 py-2 text-xs text-slate-600">
		                        <div className="truncate font-medium text-slate-800">{stripComboName(row.left)} / {stripComboName(row.right)}</div>
		                        <div className="mt-1">corr {formatNumber(row.corr, 3)}</div>
		                      </div>
		                    ))}
		                  </div>
		                </div>
		              </div>
		            ) : null}
		          </div>

		          <div className="mt-5 min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
		            <div className="mb-1 text-sm font-medium text-foreground">Train / Val / Test 过拟合诊断</div>
		            <div className="mb-3 text-xs leading-5 text-muted-foreground">
		              全因子所有 combo 方法的 Train、Val 与 2024 Test 指标对比。Test/Train 保留率越低，越说明训练期表现外推到 2024 后衰减明显。
		            </div>
		            <div className="mb-3 grid gap-3 text-sm text-slate-700 sm:grid-cols-2">
		              <div className="rounded-2xl bg-slate-50 p-3">Score Test/Train 中位保留率 {formatNumber(scoreRetentionMedian, 1)}%</div>
		              <div className="rounded-2xl bg-slate-50 p-3">IC Test/Train 中位保留率 {formatNumber(icRetentionMedian, 1)}%</div>
		            </div>
		            <div className="grid min-w-0 gap-5 xl:grid-cols-2">
		              <div className="min-w-0">
		                <div className="mb-2 text-xs font-medium text-slate-600">Score 衰减</div>
		                <div className="h-[300px]">
		                  <ResponsiveContainer width="100%" height="100%">
		                    <BarChart data={trainValTestRows} margin={{ top: 8, right: 16, bottom: 44, left: 0 }}>
		                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
		                      <XAxis dataKey="modelLabel" tick={{ fill: '#64748b', fontSize: 11 }} angle={-14} textAnchor="end" height={78} interval={0} />
		                      <YAxis allowDecimals={false} tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={formatAxisInteger} domain={overfitScoreDomain} />
			                      <Tooltip formatter={(value: number, name: string) => [formatNumber(Number(value), 1), name]} />
				                      <Legend />
				                      {trainValTestSeparators.map((item) => (
				                        <ReferenceLine key={`score-${item.modelLabel}`} x={item.modelLabel} stroke="#94a3b8" strokeDasharray="4 4" />
				                      ))}
			                      <Bar dataKey="trainScore" name="Train Score" fill="#94a3b8" radius={[5, 5, 0, 0]} />
			                      <Bar dataKey="valScore" name="Val Score" fill="#f59e0b" radius={[5, 5, 0, 0]} />
		                      <Bar dataKey="testScore" name="Test Score" fill="#0f766e" radius={[5, 5, 0, 0]} />
		                    </BarChart>
		                  </ResponsiveContainer>
		                </div>
		              </div>
		              <div className="min-w-0">
		                <div className="mb-2 text-xs font-medium text-slate-600">IC 衰减</div>
		                <div className="h-[300px]">
		                  <ResponsiveContainer width="100%" height="100%">
		                    <BarChart data={trainValTestRows} margin={{ top: 8, right: 16, bottom: 44, left: 0 }}>
		                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
		                      <XAxis dataKey="modelLabel" tick={{ fill: '#64748b', fontSize: 11 }} angle={-14} textAnchor="end" height={78} interval={0} />
		                      <YAxis tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={(value) => formatNumber(Number(value), 1)} domain={overfitIcDomain} />
			                      <Tooltip formatter={(value: number, name: string) => [formatNumber(Number(value), 2), name]} />
				                      <Legend />
				                      {trainValTestSeparators.map((item) => (
				                        <ReferenceLine key={`ic-${item.modelLabel}`} x={item.modelLabel} stroke="#94a3b8" strokeDasharray="4 4" />
				                      ))}
			                      <Bar dataKey="trainIC" name="Train IC" fill="#94a3b8" radius={[5, 5, 0, 0]} />
			                      <Bar dataKey="valIC" name="Val IC" fill="#f59e0b" radius={[5, 5, 0, 0]} />
		                      <Bar dataKey="testIC" name="Test IC" fill="#2563eb" radius={[5, 5, 0, 0]} />
		                    </BarChart>
		                  </ResponsiveContainer>
		                </div>
		              </div>
		            </div>
		          </div>

		          <div className="mt-5 min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
		            <div className="mb-3 text-sm font-medium text-foreground">全因子多方法 2024 OOS 指标</div>
	            <div className="h-[300px]">
	              <ResponsiveContainer width="100%" height="100%">
	                <BarChart data={modelComparison} margin={{ top: 8, right: 16, bottom: 36, left: 0 }}>
	                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
	                  <XAxis dataKey="modelLabel" tick={{ fill: '#64748b', fontSize: 11 }} angle={-14} textAnchor="end" height={70} interval={0} />
	                  <YAxis yAxisId="left" allowDecimals={false} tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={formatAxisInteger} domain={modelComparisonDomains.left} />
	                  <YAxis yAxisId="right" orientation="right" allowDecimals={false} tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={formatAxisInteger} domain={modelComparisonDomains.right} />
		                  <Tooltip />
		                  <Legend />
			                  <ReferenceLine yAxisId="left" y={0} stroke="#94a3b8" strokeDasharray="3 3" />
			                  {modelComparisonSeparators.map((item) => (
			                    <ReferenceLine key={item.modelLabel} yAxisId="left" x={item.modelLabel} stroke="#94a3b8" strokeDasharray="4 4" />
			                  ))}
		                  <Bar yAxisId="left" dataKey="score" name="OOS Score" fill="#0f766e" radius={[6, 6, 0, 0]} />
	                  <Bar yAxisId="right" dataKey="ic" name="OOS IC" fill="#2563eb" radius={[6, 6, 0, 0]} />
	                  <Bar yAxisId="right" dataKey="ir" name="OOS IR" fill="#f59e0b" radius={[6, 6, 0, 0]} />
	                </BarChart>
	              </ResponsiveContainer>
	            </div>
	          </div>

	          <div className="mt-5 min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
	            <div className="mb-3 text-sm font-medium text-foreground">低相关 8 因子多方法 2024 OOS 指标</div>
	            <div className="h-[300px]">
	              <ResponsiveContainer width="100%" height="100%">
	                <BarChart data={lowCorrModelComparison} margin={{ top: 8, right: 16, bottom: 36, left: 0 }}>
	                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
	                  <XAxis dataKey="modelLabel" tick={{ fill: '#64748b', fontSize: 11 }} angle={-14} textAnchor="end" height={70} interval={0} />
	                  <YAxis yAxisId="left" allowDecimals={false} tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={formatAxisInteger} domain={lowCorrModelComparisonDomains.left} />
	                  <YAxis yAxisId="right" orientation="right" allowDecimals={false} tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={formatAxisInteger} domain={lowCorrModelComparisonDomains.right} />
		                  <Tooltip />
		                  <Legend />
				                  <ReferenceLine yAxisId="left" y={0} stroke="#94a3b8" strokeDasharray="3 3" />
				                  {lowCorrModelComparisonSeparators.map((item) => (
				                    <ReferenceLine key={item.modelLabel} yAxisId="left" x={item.modelLabel} stroke="#94a3b8" strokeDasharray="4 4" />
				                  ))}
		                  <Bar yAxisId="left" dataKey="score" name="OOS Score" fill="#7c3aed" radius={[6, 6, 0, 0]} />
	                  <Bar yAxisId="right" dataKey="ic" name="OOS IC" fill="#2563eb" radius={[6, 6, 0, 0]} />
	                  <Bar yAxisId="right" dataKey="ir" name="OOS IR" fill="#f59e0b" radius={[6, 6, 0, 0]} />
	                </BarChart>
	              </ResponsiveContainer>
	            </div>
	          </div>

	          <div className="mt-5 min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
	            <div className="mb-1 text-sm font-medium text-foreground">低相关 vs 全量方法趋势对比</div>
	            <div className="mb-3 text-xs leading-5 text-muted-foreground">
	              同一方法在全量因子和低相关 8 因子上的 OOS Score / IC 变化是否同向，用来判断低相关小篮子的代表性。
	            </div>
	            <div className="grid min-w-0 gap-5 xl:grid-cols-2">
	              <div className="min-w-0">
	                <div className="mb-2 text-xs font-medium text-slate-600">Score 对比</div>
	                <div className="h-[260px]">
	                  <ResponsiveContainer width="100%" height="100%">
	                    <LineChart data={methodTrendComparison} margin={{ top: 8, right: 16, bottom: 36, left: 0 }}>
	                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
	                      <XAxis dataKey="modelLabel" tick={{ fill: '#64748b', fontSize: 11 }} angle={-14} textAnchor="end" height={70} interval={0} />
	                      <YAxis allowDecimals={false} tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={formatAxisInteger} />
		                      <Tooltip />
		                      <Legend />
			                      <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
			                      {methodTrendSeparators.map((item) => (
			                        <ReferenceLine key={`score-trend-${item.modelLabel}`} x={item.modelLabel} stroke="#94a3b8" strokeDasharray="4 4" />
			                      ))}
		                      <Line type="monotone" dataKey="fullScore" name="全量 Score" stroke="#0f766e" strokeWidth={2.2} dot={false} />
	                      <Line type="monotone" dataKey="lowScore" name="低相关 Score" stroke="#7c3aed" strokeWidth={2.2} dot={false} />
	                    </LineChart>
	                  </ResponsiveContainer>
	                </div>
	              </div>
	              <div className="min-w-0">
	                <div className="mb-2 text-xs font-medium text-slate-600">IC 对比</div>
	                <div className="h-[260px]">
	                  <ResponsiveContainer width="100%" height="100%">
	                    <LineChart data={methodTrendComparison} margin={{ top: 8, right: 16, bottom: 36, left: 0 }}>
	                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
	                      <XAxis dataKey="modelLabel" tick={{ fill: '#64748b', fontSize: 11 }} angle={-14} textAnchor="end" height={70} interval={0} />
	                      <YAxis allowDecimals={false} tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={formatAxisInteger} />
		                      <Tooltip />
		                      <Legend />
			                      <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
			                      {methodTrendSeparators.map((item) => (
			                        <ReferenceLine key={`ic-trend-${item.modelLabel}`} x={item.modelLabel} stroke="#94a3b8" strokeDasharray="4 4" />
			                      ))}
		                      <Line type="monotone" dataKey="fullIC" name="全量 IC" stroke="#2563eb" strokeWidth={2.2} dot={false} />
	                      <Line type="monotone" dataKey="lowIC" name="低相关 IC" stroke="#f59e0b" strokeWidth={2.2} dot={false} />
	                    </LineChart>
	                  </ResponsiveContainer>
	                </div>
	              </div>
	            </div>
	          </div>

		          <div className="mt-5 grid min-w-0 gap-5 xl:grid-cols-2">
		            <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
		              <div className="mb-1 text-sm font-medium text-foreground">全因子最佳 combo 输出</div>
		              <div className="mb-3 text-xs leading-5 text-muted-foreground">
		                这里展示全因子 Model Lab 的最佳 combo 输出，以及该输出与入模因子的相关性。
		              </div>
		              {latestModelLab?.best_model && latestModelLab.models?.[latestModelLab.best_model] ? (
		                (() => {
		                  const bestName = latestModelLab.best_model!;
		                  const bestPayload = latestModelLab.models![bestName];
		                  const pqPath = latestModelLab.ensemble_outputs?.[bestName] || '';
		                  const inputCorrs = (
		                    bestPayload.input_factor_correlations
		                    || latestModelLab.best_model_input_factor_correlations
		                    || []
		                  ).slice(0, 8);
		                  return (
		                    <div className="space-y-3">
		                      <div className="rounded-2xl bg-slate-50 p-3">
		                        <div className="flex items-center justify-between gap-2">
		                          <div className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">{bestName}</div>
		                          <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-medium text-emerald-700">FULL</span>
		                        </div>
		                        {pqPath ? (
		                          <div className="mt-1.5 break-all font-mono text-[11px] leading-5 text-slate-600">{pqPath.split('/').pop()}</div>
		                        ) : null}
		                        <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
		                          <span className="font-semibold text-emerald-700">IC {formatNumber(bestPayload.submit_IC ?? 0, 2)}</span>
		                          <span className="font-semibold text-emerald-700">Score {formatNumber(bestPayload.submit_Score ?? 0, 2)}</span>
		                          <span className="font-semibold text-emerald-700">IR {formatNumber(bestPayload.submit_IR ?? 0, 2)}</span>
		                          <span className="font-semibold text-emerald-700">TVR {formatNumber(getComboDisplayTvr(bestPayload), 1)}</span>
		                        </div>
		                      </div>

		                      <div className="rounded-2xl bg-slate-50 p-3">
		                        <div className="mb-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">full pq 输出 vs 输入因子相关性</div>
		                        <div className="space-y-1.5">
		                          {inputCorrs.map((item) => {
		                            const corr = Number(item.corr || 0);
		                            const absPct = Math.min(100, Math.abs(corr) * 100);
		                            const isPos = corr >= 0;
		                            return (
		                              <div key={item.run_id} className="flex items-center gap-2 text-[10px]">
		                                <div className="w-[92px] shrink-0 truncate font-mono text-slate-500">{item.run_id}</div>
		                                <div className="relative flex flex-1 items-center">
		                                  <div className="h-1.5 w-full rounded-full bg-slate-200">
		                                    <div
		                                      className={`absolute top-0 h-1.5 rounded-full ${isPos ? 'left-1/2 bg-sky-400' : 'right-1/2 bg-orange-400'}`}
		                                      style={{ width: `${absPct / 2}%` }}
		                                    />
		                                  </div>
		                                </div>
		                                <span className={`w-12 shrink-0 text-right font-semibold ${isPos ? 'text-sky-600' : 'text-orange-600'}`}>
		                                  {formatNumber(corr, 3)}
		                                </span>
		                              </div>
		                            );
		                          })}
		                        </div>
		                      </div>
		                    </div>
		                  );
		                })()
		              ) : (
		                <div className="text-sm text-muted-foreground">全因子 combo 结果完成后会显示在这里。</div>
		              )}
		            </div>

		            <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
		              <div className="mb-1 text-sm font-medium text-foreground">低相关 8 因子最佳 combo 输出</div>
		              <div className="mb-3 text-xs leading-5 text-muted-foreground">
		                这里展示低相关探索 block 的最佳 combo 输出，以及该输出与 8 个输入因子的相关性。导出的 pq 文件名会带 `lowcorr8` 后缀。
		              </div>
		              {lowCorrModelLab?.best_model && lowCorrModelLab.models?.[lowCorrModelLab.best_model] ? (
		                (() => {
		                  const bestName = lowCorrModelLab.best_model!;
		                  const bestPayload = lowCorrModelLab.models![bestName];
		                  const pqPath = lowCorrModelLab.ensemble_outputs?.[bestName] || '';
		                  const inputCorrs = (
		                    bestPayload.input_factor_correlations
		                    || lowCorrModelLab.best_model_input_factor_correlations
		                    || []
		                  ).slice(0, 8);
		                  return (
		                    <div className="space-y-3">
		                      <div className="rounded-2xl bg-slate-50 p-3">
		                        <div className="flex items-center justify-between gap-2">
		                          <div className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">{bestName}</div>
		                          <span className="rounded-full bg-violet-100 px-2 py-0.5 text-[10px] font-medium text-violet-700">LOW CORR 8</span>
		                        </div>
		                        {pqPath ? (
		                          <div className="mt-1.5 break-all font-mono text-[11px] leading-5 text-slate-600">{pqPath.split('/').pop()}</div>
		                        ) : null}
		                        <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
		                          <span className="font-semibold text-emerald-700">IC {formatNumber(bestPayload.submit_IC ?? 0, 2)}</span>
		                          <span className="font-semibold text-emerald-700">Score {formatNumber(bestPayload.submit_Score ?? 0, 2)}</span>
		                          <span className="font-semibold text-emerald-700">IR {formatNumber(bestPayload.submit_IR ?? 0, 2)}</span>
		                          <span className="font-semibold text-emerald-700">TVR {formatNumber(getComboDisplayTvr(bestPayload), 1)}</span>
		                        </div>
		                      </div>

		                      <div className="rounded-2xl bg-slate-50 p-3">
		                        <div className="mb-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">lowcorr8 pq 输出 vs 输入因子相关性</div>
		                        <div className="space-y-1.5">
		                          {inputCorrs.map((item) => {
		                            const corr = Number(item.corr || 0);
		                            const absPct = Math.min(100, Math.abs(corr) * 100);
		                            const isPos = corr >= 0;
		                            return (
		                              <div key={item.run_id} className="flex items-center gap-2 text-[10px]">
		                                <div className="w-[92px] shrink-0 truncate font-mono text-slate-500">{item.run_id}</div>
		                                <div className="relative flex flex-1 items-center">
		                                  <div className="h-1.5 w-full rounded-full bg-slate-200">
		                                    <div
		                                      className={`absolute top-0 h-1.5 rounded-full ${isPos ? 'left-1/2 bg-sky-400' : 'right-1/2 bg-orange-400'}`}
		                                      style={{ width: `${absPct / 2}%` }}
		                                    />
		                                  </div>
		                                </div>
		                                <span className={`w-12 shrink-0 text-right font-semibold ${isPos ? 'text-sky-600' : 'text-orange-600'}`}>
		                                  {formatNumber(corr, 3)}
		                                </span>
		                              </div>
		                            );
		                          })}
		                        </div>
		                      </div>
		                    </div>
		                  );
		                })()
		              ) : (
		                <div className="text-sm text-muted-foreground">低相关 8 因子探索结果完成后会显示在这里。</div>
		              )}
		            </div>
		          </div>
        </Panel>

      {selectedRunId ? <ResearchModal runId={selectedRunId} onClose={() => setSelectedRunId(null)} /> : null}
      {comboModal && (comboModal.lab === 'low' ? lowCorrModelLab : latestModelLab) ? (
        <EnsembleModal
          modelName={comboModal.modelName}
          modelLab={(comboModal.lab === 'low' ? lowCorrModelLab : latestModelLab)!}
          onOpenFactor={setSelectedRunId}
          onClose={() => setComboModal(null)}
        />
      ) : null}
      </div>
    </PageErrorBoundary>
  );
};
