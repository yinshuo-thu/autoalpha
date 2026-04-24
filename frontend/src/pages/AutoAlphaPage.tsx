import React, { useEffect, useMemo, useRef, useState } from 'react';
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
  ReferenceArea,
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
  avg_sharpe: number;
  total_pnl: number;
  gross_pnl?: number;
  total_fee?: number;
  max_drawdown: number;
  hit_ratio: number;
  avg_turnover?: number;
  cumulative_curve: ModelLabPoint[];
  drawdown_curve?: ModelLabPoint[];
  gross_cumulative_curve?: ModelLabPoint[];
  fee_cumulative_curve?: ModelLabPoint[];
  daily_pnl_curve: ModelLabPoint[];
  prediction_comparison_curve?: Array<{
    date: string;
    mean_prediction: number;
    mean_return: number;
    mean_prediction_aligned?: number;
    predicted_spread: number;
    realized_spread: number;
    predicted_spread_aligned?: number;
    window_id?: number;
    test_start?: string;
    test_end?: string;
  }>;
  top_features: Array<{
    factor: string;
    importance: number;
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
  window_count: number;
  train_days: number;
  test_days: number;
  step_days: number;
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
}

interface ModelLabPayload {
  latest: ModelLabSummary | null;
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

interface FeatureConsensusPoint {
  factor: string;
  shortLabel: string;
  consensusImportance: number;
  avgImportance: number;
  supportCount: number;
  supportRatio: number;
  models: string[];
  formula?: string;
  score?: number;
  ic?: number;
  dominantTrack: string;
  dominantTrackLabel: string;
  dominantTrackSupportCount: number;
  dominantTrackConsensus: number;
  familyFeatureCount: number;
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

interface FactorCorrelationsData {
  run_ids: string[];
  labels: string[];
  matrix: number[][];
  updated_at: string;
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

const MODEL_COLORS = ['#0f766e', '#2563eb', '#9333ea', '#dc2626', '#0891b2', '#ca8a04', '#4f46e5', '#16a34a'];
const MODEL_COMPARISON_CHART_MARGIN = { top: 12, right: 32, bottom: 48, left: 8 };
const MODEL_COMPARISON_LEFT_AXIS_WIDTH = 56;
const MODEL_COMPARISON_RIGHT_AXIS_WIDTH = 56;

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

function formatRollingDate(value?: string) {
  if (!value) return '--';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return truncate(value, 10);
  return date.toLocaleDateString('zh-CN', {
    year: '2-digit',
    month: '2-digit',
    day: '2-digit',
  });
}

function truncate(text: string, length: number) {
  if (!text) return '—';
  return text.length > length ? `${text.slice(0, length)}…` : text;
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

function mergeModelCurves(curvesByModel: Record<string, ModelLabPoint[]>) {
  const merged = new Map<string, Record<string, string | number>>();
  Object.entries(curvesByModel).forEach(([modelName, points]) => {
    points.forEach((point) => {
      const row = merged.get(point.date) || { date: point.date };
      row[modelName] = point.value;
      merged.set(point.date, row);
    });
  });
  return Array.from(merged.values()).sort((a, b) => String(a.date).localeCompare(String(b.date)));
}

function buildMergedModelCurves(models: Record<string, ModelLabModelSummary> | undefined) {
  const curvesByModel = Object.fromEntries(
    Object.entries(models || {}).map(([modelName, payload]) => [modelName, payload.cumulative_curve || []])
  );
  return mergeModelCurves(curvesByModel);
}

function buildMergedDrawdownCurves(models: Record<string, ModelLabModelSummary> | undefined) {
  const curvesByModel = Object.fromEntries(
    Object.entries(models || {}).map(([modelName, payload]) => {
      const drawdownCurve = payload.drawdown_curve?.length
        ? payload.drawdown_curve
        : (() => {
            let peak = Number.NEGATIVE_INFINITY;
            return (payload.cumulative_curve || []).map((point) => {
              peak = Math.max(peak, Number(point.value || 0));
              return { date: point.date, value: Number(point.value || 0) - peak };
            });
          })();
      return [modelName, drawdownCurve];
    })
  );
  return mergeModelCurves(curvesByModel);
}

function alignSeriesMeanStd(source: number[], target: number[]) {
  if (source.length !== target.length || source.length === 0) return source;
  const cleanSource = source.map((value) => (Number.isFinite(value) ? value : 0));
  const cleanTarget = target.map((value) => (Number.isFinite(value) ? value : 0));
  const sourceMean = cleanSource.reduce((sum, value) => sum + value, 0) / cleanSource.length;
  const targetMean = cleanTarget.reduce((sum, value) => sum + value, 0) / cleanTarget.length;
  const sourceVariance = cleanSource.reduce((sum, value) => sum + ((value - sourceMean) ** 2), 0) / cleanSource.length;
  const targetVariance = cleanTarget.reduce((sum, value) => sum + ((value - targetMean) ** 2), 0) / cleanTarget.length;
  const sourceStd = Math.sqrt(sourceVariance);
  const targetStd = Math.sqrt(targetVariance);
  if (!Number.isFinite(sourceStd) || sourceStd <= 0) {
    return cleanSource.map(() => targetMean);
  }
  const scale = Number.isFinite(targetStd) && targetStd > 0 ? targetStd / sourceStd : 0;
  return cleanSource.map((value) => ((value - sourceMean) * scale) + targetMean);
}

function buildPredictionComparisonCurve(model?: ModelLabModelSummary) {
  const points = [...(model?.prediction_comparison_curve || [])].sort((a, b) => String(a.date).localeCompare(String(b.date)));
  if (points.length === 0) return [];
  const alignedSpread = alignSeriesMeanStd(
    points.map((point) => Number(point.predicted_spread || 0)),
    points.map((point) => Number(point.realized_spread || 0))
  );
  const alignedMean = alignSeriesMeanStd(
    points.map((point) => Number(point.mean_prediction || 0)),
    points.map((point) => Number(point.mean_return || 0))
  );
  return points.map((point, index) => ({
    ...point,
    predicted_spread_display: Number(
      point.predicted_spread_aligned ?? alignedSpread[index] ?? point.predicted_spread ?? 0
    ),
    mean_prediction_display: Number(
      point.mean_prediction_aligned ?? alignedMean[index] ?? point.mean_prediction ?? 0
    ),
  }));
}

function compactFeatureLabel(value: string) {
  if (!value) return '—';
  return value.length <= 20 ? value : `${value.slice(0, 8)}…${value.slice(-6)}`;
}

const FEATURE_TRACK_LABELS: Record<string, string> = {
  raw: '原始值',
  csz: '截面 Z 分数',
  rank: '截面 Rank',
  lag1: '前 1 期',
  diff1: '1 期变化',
  mom2: '2 期动量',
};

function parseModelFeatureName(value: string) {
  const text = String(value || '').trim();
  if (!text) {
    return {
      rawFeature: '',
      baseFactor: '',
      track: 'raw',
      trackLabel: FEATURE_TRACK_LABELS.raw,
    };
  }
  const [baseFactor, ...suffixParts] = text.split('__');
  const track = suffixParts.join('__') || 'raw';
  return {
    rawFeature: text,
    baseFactor,
    track,
    trackLabel: FEATURE_TRACK_LABELS[track] || track.toUpperCase(),
  };
}

function featureConsensusColor(supportRatio: number) {
  if (supportRatio >= 0.99) return '#0f766e';
  if (supportRatio >= 0.66) return '#2563eb';
  if (supportRatio >= 0.5) return '#7c3aed';
  return '#94a3b8';
}

function buildFeatureConsensus(
  models: Record<string, ModelLabModelSummary> | undefined,
  selectedFactors: ModelLabSummary['selected_factors'] | undefined
) {
  const modelEntries = Object.entries(models || {});
  if (modelEntries.length === 0) return [];

  const selectedMap = new Map(
    (selectedFactors || []).map((factor) => [
      factor.run_id,
      { formula: factor.formula, score: factor.score, ic: factor.ic },
    ])
  );

  const aggregated = new Map<
    string,
    {
      factor: string;
      consensusImportance: number;
      totalImportance: number;
      supportCount: number;
      models: string[];
      formula?: string;
      score?: number;
      ic?: number;
      tracks: Map<
        string,
        {
          track: string;
          label: string;
          consensusImportance: number;
          totalImportance: number;
          supportCount: number;
          models: string[];
        }
      >;
    }
  >();

  modelEntries.forEach(([modelName, payload]) => {
    const performanceWeight = Math.max(
      0.7,
      1 + (Math.max(payload.avg_daily_ic, 0) * 10) + (Math.max(payload.avg_sharpe, 0) * 0.06)
    );

    (payload.top_features || []).slice(0, 8).forEach((item, index) => {
      const parsed = parseModelFeatureName(String(item.factor || ''));
      const factor = parsed.baseFactor;
      if (!factor) return;

      const meta = selectedMap.get(factor);
      const rankBoost = 1 + ((8 - index) / 10);
      const importance = Number(item.importance || 0);
      const weightedImportance = importance * performanceWeight * rankBoost;
      const current = aggregated.get(factor) || {
        factor,
        consensusImportance: 0,
        totalImportance: 0,
        supportCount: 0,
        models: [] as string[],
        formula: meta?.formula,
        score: meta?.score,
        ic: meta?.ic,
        tracks: new Map<
          string,
          {
            track: string;
            label: string;
            consensusImportance: number;
            totalImportance: number;
            supportCount: number;
            models: string[];
          }
        >(),
      };

      current.consensusImportance += weightedImportance;
      current.totalImportance += importance;

      if (!current.models.includes(modelName)) {
        current.models.push(modelName);
        current.supportCount += 1;
      }

      if (!current.formula && meta?.formula) current.formula = meta.formula;
      if (current.score === undefined && meta?.score !== undefined) current.score = meta.score;
      if (current.ic === undefined && meta?.ic !== undefined) current.ic = meta.ic;

      const trackCurrent = current.tracks.get(parsed.track) || {
        track: parsed.track,
        label: parsed.trackLabel,
        consensusImportance: 0,
        totalImportance: 0,
        supportCount: 0,
        models: [] as string[],
      };
      trackCurrent.consensusImportance += weightedImportance;
      trackCurrent.totalImportance += importance;
      if (!trackCurrent.models.includes(modelName)) {
        trackCurrent.models.push(modelName);
        trackCurrent.supportCount += 1;
      }
      current.tracks.set(parsed.track, trackCurrent);

      aggregated.set(factor, current);
    });
  });

  return Array.from(aggregated.values())
    .map((item) => {
      const orderedTracks = Array.from(item.tracks.values()).sort((a, b) => {
        if (b.consensusImportance !== a.consensusImportance) return b.consensusImportance - a.consensusImportance;
        if (b.supportCount !== a.supportCount) return b.supportCount - a.supportCount;
        return b.totalImportance - a.totalImportance;
      });
      const dominantTrack = orderedTracks[0];
      return {
        factor: item.factor,
        shortLabel: compactFeatureLabel(item.factor),
        consensusImportance: Number(item.consensusImportance.toFixed(3)),
        avgImportance: Number((item.totalImportance / Math.max(item.supportCount, 1)).toFixed(3)),
        supportCount: item.supportCount,
        supportRatio: item.supportCount / modelEntries.length,
        models: item.models,
        formula: item.formula,
        score: item.score,
        ic: item.ic,
        dominantTrack: dominantTrack?.track || 'raw',
        dominantTrackLabel: dominantTrack?.label || FEATURE_TRACK_LABELS.raw,
        dominantTrackSupportCount: dominantTrack?.supportCount || 0,
        dominantTrackConsensus: Number((dominantTrack?.consensusImportance || 0).toFixed(3)),
        familyFeatureCount: orderedTracks.length,
      };
    })
    .sort((a, b) => {
      if (b.consensusImportance !== a.consensusImportance) return b.consensusImportance - a.consensusImportance;
      if (b.supportCount !== a.supportCount) return b.supportCount - a.supportCount;
      return b.avgImportance - a.avgImportance;
    })
    .slice(0, 8);
}

function niceUpperBound(value: number, minUpper: number, padding = 1.25) {
  const raw = Math.max(value * padding, minUpper);
  if (!Number.isFinite(raw) || raw <= 0) return minUpper;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalized = raw / magnitude;
  const step = normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return step * magnitude;
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
  const points = ordered.map((factor, index) => {
    if (factor.PassGates) passing += 1;
    bestScore = Math.max(bestScore, Number(factor.Score || 0));
    const timestamp = factor.created_at || '';
    return {
      index: index + 1,
      timestamp,
      label: timestamp ? timestamp.replace('T', ' ').slice(0, 16) : `#${index + 1}`,
      tested: index + 1,
      passing,
      best_score: Number(bestScore.toFixed(2)),
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

  const apiIsCurrent =
    apiLast &&
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

const LogPanel = ({ logs }: { logs: string[] }) => {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [logs]);

  const colorLine = (line: string) => {
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
            {line}
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

function getSelectedFactorCorrelations(
  selectedRunIds: string[],
  corrData: FactorCorrelationsData,
): Array<{ a: string; b: string; corr: number }> {
  const pairs: Array<{ a: string; b: string; corr: number }> = [];
  for (let i = 0; i < selectedRunIds.length; i++) {
    const idxA = corrData.run_ids.indexOf(selectedRunIds[i]);
    if (idxA === -1) continue;
    for (let j = i + 1; j < selectedRunIds.length; j++) {
      const idxB = corrData.run_ids.indexOf(selectedRunIds[j]);
      if (idxB === -1) continue;
      const corr = corrData.matrix[idxA]?.[idxB];
      if (corr !== undefined && corr !== null && Number.isFinite(corr)) {
        pairs.push({ a: selectedRunIds[i], b: selectedRunIds[j], corr });
      }
    }
  }
  return pairs.sort((a, b) => Math.abs(b.corr) - Math.abs(a.corr));
}

const EnsembleModal = ({
  modelName,
  modelLab,
  factorCorrelations,
  onOpenFactor,
  onClose,
}: {
  modelName: string;
  modelLab: ModelLabSummary;
  factorCorrelations: FactorCorrelationsData;
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
  const selectedRunIds = (modelLab.selected_factors || []).map((f) => f.run_id);
  const correlationPairs = getSelectedFactorCorrelations(selectedRunIds, factorCorrelations);
  const isBest = modelName === modelLab.best_model;

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
      <div className="max-h-[90vh] w-full max-w-xl overflow-y-auto rounded-3xl border border-border/60 bg-white p-6 shadow-2xl">
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

        <div className="mb-4 rounded-2xl bg-slate-50 p-3">
          <div className="mb-1 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">输出文件路径</div>
          <div className="break-all font-mono text-[11px] leading-6 text-slate-700">{path || '--'}</div>
          <div className="mt-1 text-[10px] text-muted-foreground">仅最佳模型产出 pq 文件；可按 submission 格式提交。</div>
        </div>

        {modelPayload ? (
          <div className="mb-4 rounded-2xl bg-slate-50 p-3">
            <div className="mb-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">模型表现（滚动均值）</div>
            <div className="grid grid-cols-3 gap-y-2 text-[11px] text-slate-600">
              <div>IC {formatNumber(modelPayload.avg_daily_ic, 4)}</div>
              <div>RankIC {formatNumber(modelPayload.avg_daily_rank_ic, 4)}</div>
              <div>Sharpe {formatNumber(modelPayload.avg_sharpe, 2)}</div>
              <div>TVR {formatNumber(modelPayload.avg_turnover ?? 0, 3)}</div>
              <div>MaxDD {formatNumber(modelPayload.max_drawdown, 3)}</div>
              <div>Hit {formatPercent(modelPayload.hit_ratio * 100)}</div>
            </div>
          </div>
        ) : null}

        {modelPayload?.top_features?.length ? (
          <div className="mb-4 rounded-2xl bg-slate-50 p-3">
            <div className="mb-3 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">因子贡献权重</div>
            <div className="space-y-2.5">
              {modelPayload.top_features.slice(0, 8).map((item) => {
                const maxImp = modelPayload.top_features[0]?.importance || 1;
                const pct = Math.min(100, (item.importance / maxImp) * 100);
                return (
                  <div key={item.factor}>
                    <div className="mb-1 flex items-center justify-between gap-2">
                      <button
                        onClick={() => { onOpenFactor(item.factor); onClose(); }}
                        className="min-w-0 truncate text-left font-mono text-[11px] text-sky-700 underline decoration-sky-400 underline-offset-4 hover:text-sky-900"
                      >
                        {item.factor}
                      </button>
                      <span className="shrink-0 text-[11px] text-slate-600">{formatNumber(item.importance, 3)}</span>
                    </div>
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
                      <div className="h-1.5 rounded-full bg-sky-400 transition-all" style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : null}

        {correlationPairs.length > 0 ? (
          <div className="mb-4 rounded-2xl bg-slate-50 p-3">
            <div className="mb-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">入模因子相关性（绝对值降序，前 6 对）</div>
            <div className="space-y-1.5">
              {correlationPairs.slice(0, 6).map(({ a, b, corr }) => (
                <div key={`${a}-${b}`} className="flex items-center gap-2 text-[11px]">
                  <span className="w-[40%] truncate font-mono text-slate-600">{a}</span>
                  <span className="text-muted-foreground">↔</span>
                  <span className="w-[40%] truncate font-mono text-slate-600">{b}</span>
                  <span className={`ml-auto shrink-0 font-semibold ${Math.abs(corr) > 0.7 ? 'text-amber-600' : Math.abs(corr) > 0.4 ? 'text-sky-600' : 'text-emerald-600'}`}>
                    {corr.toFixed(3)}
                  </span>
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
  const [factorCorrelations, setFactorCorrelations] = useState<FactorCorrelationsData>({ run_ids: [], labels: [], matrix: [], updated_at: '' });
  const [ensembleModalModel, setEnsembleModalModel] = useState<string | null>(null);
  const [rounds, setRounds] = useState(10);
  const [ideas, setIdeas] = useState(4);
  const [days, setDays] = useState(0);
  const [targetValid, setTargetValid] = useState(0);
  const [promptTitle, setPromptTitle] = useState('');
  const [promptInput, setPromptInput] = useState('');
  const [promptBusy, setPromptBusy] = useState(false);
  const [manualBusy, setManualBusy] = useState(false);
  const [manualFactor, setManualFactor] = useState<ManualFactor | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [pageMessage, setPageMessage] = useState('');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let mounted = true;

    const loadAll = async () => {
      const [statusResult, knowledgeResult, balanceResult, modelLabResult, inspirationResult, corrResult] = await Promise.allSettled([
        fetchJson<LoopStatus>('/api/autoalpha/loop/status'),
        fetchJson<KnowledgePayload>('/api/autoalpha/knowledge'),
        fetchJson<BalancePayload>('/api/autoalpha/balance'),
        fetchJson<ModelLabPayload>('/api/autoalpha/model-lab'),
        fetchJson<InspirationPayload>('/api/autoalpha/inspirations'),
        fetchJson<FactorCorrelationsData>('/api/autoalpha/factor-correlations'),
      ]);

      if (!mounted) return;
      if (statusResult.status === 'fulfilled') setStatus(statusResult.value);
      if (knowledgeResult.status === 'fulfilled') setKnowledge(knowledgeResult.value);
      if (balanceResult.status === 'fulfilled') setBalance(balanceResult.value);
      if (modelLabResult.status === 'fulfilled') setModelLab(modelLabResult.value);
      if (inspirationResult.status === 'fulfilled') setInspirations(inspirationResult.value);
      if (corrResult.status === 'fulfilled') setFactorCorrelations(corrResult.value);
    };

    loadAll();
    pollRef.current = setInterval(loadAll, 4000);

    fetchJson<RuntimeConfigPayload>('/api/system/config')
      .then((cfg) => {
        const env = cfg.env || {};
        setRounds(Number(env.AUTOALPHA_DEFAULT_ROUNDS || 10));
        setIdeas(Number(env.AUTOALPHA_DEFAULT_IDEAS || 4));
        setDays(Number(env.AUTOALPHA_DEFAULT_DAYS || 0));
        setTargetValid(Number(env.AUTOALPHA_DEFAULT_TARGET_VALID || 0));
      })
      .catch(() => {});

    return () => {
      mounted = false;
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const isRunning = status?.is_running ?? false;
  const factors = knowledge?.factors ?? [];
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
  const passingAxisMax = niceUpperBound(Math.max(...progressChartPoints.map((point) => point.passing), 0), 60);
  const passRateAxisMax = niceUpperBound(Math.max(...progressChartPoints.map((point) => point.pass_rate), 0), 5);
  const efficiencyAxisMax = niceUpperBound(Math.max(...progressChartPoints.map((point) => point.generation_efficiency), 0), 5);
  const latestModelLab = modelLab?.latest ?? null;
  const modelLabCurve = buildMergedModelCurves(latestModelLab?.models);
  const modelLabDrawdownCurve = buildMergedDrawdownCurves(latestModelLab?.models);
  const bestModelPayload = latestModelLab?.best_model ? latestModelLab.models?.[latestModelLab.best_model] : undefined;
  const predictionComparisonCurve = buildPredictionComparisonCurve(bestModelPayload);
  const rollingWindowBands = latestModelLab?.windows || [];
  const featureConsensus = useMemo(
    () => buildFeatureConsensus(latestModelLab?.models, latestModelLab?.selected_factors),
    [latestModelLab]
  );
  const featureConsensusAxisMax = niceUpperBound(
    Math.max(...featureConsensus.map((item) => item.consensusImportance), 0),
    1
  );
  const topConsensusFeature = featureConsensus[0];
  const widestSupportFeature = [...featureConsensus].sort((a, b) => b.supportCount - a.supportCount || b.consensusImportance - a.consensusImportance)[0];
  const richestTrackFeature = [...featureConsensus].sort(
    (a, b) => b.familyFeatureCount - a.familyFeatureCount || b.consensusImportance - a.consensusImportance
  )[0];
  const bestModelTopFeature = bestModelPayload?.top_features?.[0];
  const bestModelTopFeatureParsed = parseModelFeatureName(String(bestModelTopFeature?.factor || ''));
  const bestModelTopFeatureFamily = featureConsensus.find((item) => item.factor === bestModelTopFeatureParsed.baseFactor);
  const modelCount = Object.keys(latestModelLab?.models || {}).length;

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
    1
  );

  const modelComparison = Object.entries(latestModelLab?.models || {}).map(([modelName, payload]) => ({
    model: modelName,
    ic: payload.avg_daily_ic,
    rankIc: payload.avg_daily_rank_ic,
    sharpe: payload.avg_sharpe,
    pnl: payload.total_pnl,
  }));

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
        <div className="grid min-w-0 gap-4 md:grid-cols-2 xl:grid-cols-5">
          <StatCard label="已测试因子" value={String(status?.total_tested ?? 0)} helper="累计知识库记录" />
          <StatCard label="通过 Gate" value={String(status?.total_passing ?? 0)} helper={`通过率 ${formatPercent(knowledge?.pass_rate ?? 0)}`} accent="bg-emerald-50" />
          <StatCard label="最佳 Score" value={formatNumber(status?.best_score ?? 0, 2)} helper="按云端一致口径显示" accent="bg-sky-50" />
      <StatCard label="Ideas" value={String(inspirations?.count ?? 0)} helper="Manual / Paper / LLM / Future" accent="bg-violet-50" />
          <StatCard label="Rolling 窗口" value={String(latestModelLab?.window_count ?? 0)} helper={latestModelLab ? `${latestModelLab.best_model} 最优` : '等待实验结果'} accent="bg-amber-50" />
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

      <div className="grid min-w-0 items-start gap-6 xl:grid-cols-2">
        <Panel
          title="额度包与成本"
          subtitle="本页按第三方订阅接口返回的原始额度口径展示；换算关系为 $2 额度约对应 ¥1 实际中转消费。"
          right={<div className={`rounded-full bg-emerald-500/10 px-4 py-2 text-sm font-medium ${quotaTone(balance?.quota_status ?? 'healthy')}`}>{balance?.quota_status ?? 'healthy'}</div>}
          className="h-full"
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

        <Panel title="研究进程综述" className="h-full">
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
              <div className="mb-3 text-sm font-medium text-foreground">最佳 Score 趋势</div>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={progressPoints}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="timestamp" tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={formatAxisTime} minTickGap={28} />
                    <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
                    <Tooltip />
                    <Line type="monotone" dataKey="best_score" stroke={COLORS.bestScore} strokeWidth={3} dot={false} />
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
          </div>
        </Panel>

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

        <Panel title="循环控制与实时日志" subtitle="直接启动全量数据挖掘；rounds=0 表示持续运行，直到达到目标有效因子数或手动停止。" className="h-full">
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
              <input type="number" min={0} max={1000} value={targetValid} onChange={(event) => setTargetValid(Math.max(0, Number(event.target.value)))} disabled={isRunning} className="mt-3 w-full rounded-2xl border border-border/60 bg-slate-50 px-3 py-2 text-base outline-none disabled:opacity-50" />
              <div className="mt-2 text-[11px] text-muted-foreground">0 表示不设目标</div>
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
            <StatCard label="最近更新" value={formatDateTime(status?.updated_at)} helper={status?.pid ? `PID ${status.pid}` : '状态轮询间隔 4 秒'} valueClassName="text-xl" />
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

      <div className="grid min-w-0 gap-6">
        <Panel title="Rolling Model Lab" subtitle="基于当前已获得的有效因子做滚动训练实验：支持 1 个或更多有效因子，半年训练、半年测试、继续向后滚动，并输出线性模型 / LightGBM 的预测效果、PnL 曲线和整体因子 parquet。">
          <div className="grid min-w-0 gap-4 md:grid-cols-2 2xl:grid-cols-4">
            <StatCard label="Run ID" value={latestModelLab?.run_id || '--'} helper={latestModelLab ? formatDateTime(latestModelLab.created_at) : '还没有实验'} valueClassName="text-lg" />
            <StatCard label="选中因子数" value={String(latestModelLab?.selected_factor_count ?? 0)} helper={`目标 ${latestModelLab?.target_valid_count ?? 0}`} accent="bg-violet-50" />
            <StatCard label="滚动窗口数" value={String(latestModelLab?.window_count ?? 0)} helper={latestModelLab ? `${latestModelLab.train_days}/${latestModelLab.test_days}/${latestModelLab.step_days}` : 'train/test/step'} accent="bg-sky-50" />
            <StatCard label="最优模型" value={latestModelLab?.best_model || '--'} helper="线性 / LGB 竞赛" accent="bg-emerald-50" />
          </div>

          <div className="mt-5 grid min-w-0 gap-5">
            <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
              <div className="mb-1 text-sm font-medium text-foreground">真实策略累计 PnL 曲线</div>
              <div className="mb-3 text-xs leading-5 text-muted-foreground">
                这里改成了和预测多空分组一致的真实多空组合口径：每日做多预测最高分组、做空最低分组，并在上方同步展示对应回撤。
              </div>
              <div className="space-y-3">
                <div className="h-[150px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart syncId="model-lab-pnl" data={modelLabDrawdownCurve} margin={{ top: 8, right: 28, bottom: 4, left: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="date" hide />
                      <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
                      <Tooltip labelFormatter={(value) => formatRollingDate(String(value))} />
                      <Legend />
                      {Object.keys(latestModelLab?.models || {}).map((modelName, index) => (
                        <Line
                          key={`dd-${modelName}`}
                          type="monotone"
                          dataKey={modelName}
                          name={`${modelName} 回撤`}
                          stroke={MODEL_COLORS[index % MODEL_COLORS.length]}
                          strokeWidth={2}
                          dot={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart syncId="model-lab-pnl" data={modelLabCurve} margin={{ top: 12, right: 28, bottom: 18, left: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={formatRollingDate} minTickGap={28} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
                      <Tooltip labelFormatter={(value) => formatRollingDate(String(value))} />
                      <Legend />
                      {Object.keys(latestModelLab?.models || {}).map((modelName, index) => (
                        <Line
                          key={modelName}
                          type="monotone"
                          dataKey={modelName}
                          stroke={MODEL_COLORS[index % MODEL_COLORS.length]}
                          strokeWidth={2.5}
                          dot={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
              <div className="mb-1 text-sm font-medium text-foreground">Rolling 预测时序对比</div>
              <div className="mb-3 text-xs leading-5 text-muted-foreground">
                预测多空差已按实际多空收益做均值与波动对齐，背景分段覆盖全部 rolling test 窗口。
              </div>
              <div className="h-[420px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={predictionComparisonCurve} margin={{ top: 12, right: 28, bottom: 18, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={formatRollingDate} minTickGap={28} />
                    <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
                    <Tooltip
                      labelFormatter={(value) => formatRollingDate(String(value))}
                      formatter={(value: number, name: string) => [
                        formatNumber(Number(value), name === '标的平均收益' ? 4 : 4),
                        name,
                      ]}
                    />
                    <Legend />
                    {rollingWindowBands.map((window, index) => (
                      <ReferenceArea
                        key={`window-${window.window_id}`}
                        x1={window.test_start}
                        x2={window.test_end}
                        fill={index % 2 === 0 ? '#dbeafe' : '#dcfce7'}
                        fillOpacity={0.15}
                        strokeOpacity={0}
                      />
                    ))}
                    <Line type="monotone" dataKey="predicted_spread_display" name="预测多空差" stroke={COLORS.pnlB} strokeWidth={2.5} dot={false} />
                    <Line type="monotone" dataKey="realized_spread" name="实际多空收益" stroke={COLORS.pnlA} strokeWidth={2.5} dot={false} />
                    <Line type="monotone" dataKey="mean_return" name="标的平均收益" stroke="#94a3b8" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
              <div className="mb-1 text-sm font-medium text-foreground">模型 IC 对比</div>
              <div className="mb-3 text-xs leading-5 text-muted-foreground">
                这里只保留模型间的 Avg IC 和 Avg Rank IC，对比信号强度与排序稳定性。
              </div>
              <div className="h-[380px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={modelComparison} margin={MODEL_COMPARISON_CHART_MARGIN}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="model" tick={{ fill: '#64748b', fontSize: 11 }} angle={-12} textAnchor="end" height={54} interval={0} />
                    <YAxis yAxisId="ic" width={MODEL_COMPARISON_LEFT_AXIS_WIDTH} tick={{ fill: '#64748b', fontSize: 11 }} />
                    <YAxis
                      yAxisId="alignment"
                      orientation="right"
                      width={MODEL_COMPARISON_RIGHT_AXIS_WIDTH}
                      domain={[0, 1]}
                      tick={false}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip />
                    <Legend />
                    <Bar yAxisId="ic" dataKey="ic" name="Avg IC" fill={COLORS.ic} radius={[6, 6, 0, 0]} />
                    <Bar yAxisId="ic" dataKey="rankIc" name="Avg Rank IC" fill={COLORS.histogram} radius={[6, 6, 0, 0]} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
              <div className="mb-3 text-sm font-medium text-foreground">Sharpe / PnL 对比</div>
              <div className="h-[360px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={modelComparison} margin={MODEL_COMPARISON_CHART_MARGIN}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="model" tick={{ fill: '#64748b', fontSize: 11 }} angle={-12} textAnchor="end" height={54} interval={0} />
                    <YAxis yAxisId="sharpe" width={MODEL_COMPARISON_LEFT_AXIS_WIDTH} tick={{ fill: '#64748b', fontSize: 11 }} />
                    <YAxis yAxisId="pnl" orientation="right" width={MODEL_COMPARISON_RIGHT_AXIS_WIDTH} tick={{ fill: '#64748b', fontSize: 11 }} />
                    <Tooltip />
                    <Legend />
                    <Bar yAxisId="sharpe" dataKey="sharpe" name="Sharpe" fill={COLORS.sharpe} radius={[6, 6, 0, 0]} />
                    <Bar yAxisId="pnl" dataKey="pnl" name="Net PnL" fill={COLORS.netPnl} radius={[6, 6, 0, 0]} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="mt-5 grid min-w-0 items-start gap-5 lg:grid-cols-2">
            {/* ── Cell 1: 跨模型特征重要性共识 (top-left) ─────────── */}
            <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
              <div className="mb-1 flex items-center gap-2 text-sm font-medium text-foreground">
                <BrainCircuit className="h-4 w-4 text-violet-500" />
                跨模型特征重要性共识
              </div>
              <div className="mb-4 text-xs leading-5 text-muted-foreground">
                先把 `run_id / run_id__rank / run_id__csz / run_id__lag1` 聚回原始因子家族，再结合各模型 top features 做轻度表现加权，筛出同时"高重要性 + 多模型重复出现"的关键入模因子。
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                <div className="rounded-2xl bg-slate-50 p-3">
                  <div className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">跨模型最重要</div>
                  <div className="mt-2 break-all text-sm font-semibold text-foreground">
                    {topConsensusFeature ? topConsensusFeature.factor : '--'}
                  </div>
                  <div className="mt-1 text-[11px] text-muted-foreground">
                    共识分 {topConsensusFeature ? formatNumber(topConsensusFeature.consensusImportance, 2) : '--'}
                  </div>
                  <div className="mt-1 text-[11px] text-slate-500">
                    主导轨道 {topConsensusFeature?.dominantTrackLabel || '--'}
                  </div>
                </div>
                <div className="rounded-2xl bg-slate-50 p-3">
                  <div className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">覆盖模型最多</div>
                  <div className="mt-2 break-all text-sm font-semibold text-foreground">
                    {widestSupportFeature ? widestSupportFeature.factor : '--'}
                  </div>
                  <div className="mt-1 text-[11px] text-muted-foreground">
                    {widestSupportFeature ? `${widestSupportFeature.supportCount} 个模型同时使用` : '--'}
                  </div>
                  <div className="mt-1 text-[11px] text-slate-500">
                    代表轨道 {widestSupportFeature?.dominantTrackLabel || '--'}
                  </div>
                </div>
                <div className="rounded-2xl bg-slate-50 p-3">
                  <div className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">最佳模型首要家族</div>
                  <div className="mt-2 break-all text-sm font-semibold text-foreground">
                    {bestModelTopFeatureFamily?.factor || bestModelTopFeatureParsed.baseFactor || '--'}
                  </div>
                  <div className="mt-1 text-[11px] text-muted-foreground">
                    {latestModelLab?.best_model || '--'} · importance {bestModelTopFeature ? formatNumber(bestModelTopFeature.importance, 2) : '--'}
                  </div>
                  <div className="mt-1 text-[11px] text-slate-500">
                    当前轨道 {bestModelTopFeature ? bestModelTopFeatureParsed.trackLabel : '--'}
                  </div>
                </div>
                <div className="rounded-2xl bg-slate-50 p-3">
                  <div className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">衍生轨道最丰富</div>
                  <div className="mt-2 break-all text-sm font-semibold text-foreground">
                    {richestTrackFeature?.factor || '--'}
                  </div>
                  <div className="mt-1 text-[11px] text-muted-foreground">
                    {richestTrackFeature ? `${richestTrackFeature.familyFeatureCount} 条轨道同时进入模型` : '--'}
                  </div>
                  <div className="mt-1 text-[11px] text-slate-500">
                    主导轨道 {richestTrackFeature?.dominantTrackLabel || '--'}
                  </div>
                </div>
              </div>

              <div className="mt-4 grid gap-3 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
                <div className="rounded-2xl border border-border/50 bg-white/80 p-3 text-[11px] leading-6 text-slate-600">
                  <span className="font-medium text-slate-800">共识分</span>
                  {` = Σ(单模型 importance × 模型表现权重 × 排名加成)。`}
                  {` 模型表现权重≈max(0.7, 1 + Avg IC×10 + Sharpe×0.06)，排名加成会让 top1 比 top8 更重。`}
                  {` 所以它不是单模型 importance，而是"重要性 × 覆盖度 × 模型质量"的家族级汇总分。`}
                </div>
                <div className="rounded-2xl border border-border/50 bg-white/80 p-3 text-[11px] leading-6 text-slate-600">
                  <span className="font-medium text-slate-800">颜色只表示覆盖模型比例</span>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <span className="rounded-full px-2 py-1 text-white" style={{ backgroundColor: '#0f766e' }}>青绿 = {modelCount || 0}/{modelCount || 0} 模型</span>
                    <span className="rounded-full px-2 py-1 text-white" style={{ backgroundColor: '#2563eb' }}>蓝 = 覆盖至少 2/3</span>
                    <span className="rounded-full px-2 py-1 text-white" style={{ backgroundColor: '#7c3aed' }}>紫 = 覆盖至少 1/2</span>
                    <span className="rounded-full bg-slate-400 px-2 py-1 text-white">灰 = 只在少数模型出现</span>
                  </div>
                </div>
              </div>

              <div className="mt-4 h-[340px]">
                {featureConsensus.length ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={[...featureConsensus].reverse()}
                      layout="vertical"
                      margin={{ top: 8, right: 20, bottom: 8, left: 10 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis type="number" domain={[0, featureConsensusAxisMax]} tick={{ fill: '#64748b', fontSize: 11 }} />
                      <YAxis type="category" dataKey="shortLabel" width={92} tick={{ fill: '#64748b', fontSize: 11 }} />
                      <Tooltip
                        formatter={(value: number) => [formatNumber(Number(value), 3), '共识分']}
                        labelFormatter={(label, payload) => {
                          const item = payload?.[0]?.payload as FeatureConsensusPoint | undefined;
                          if (!item) return String(label);
                          return `${item.factor} | ${item.models.length}/${modelCount || item.models.length} 模型 | ${item.dominantTrackLabel}`;
                        }}
                      />
                      <Bar dataKey="consensusImportance" radius={[0, 8, 8, 0]}>
                        {([...featureConsensus].reverse()).map((item) => (
                          <Cell key={item.factor} fill={featureConsensusColor(item.supportRatio)} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-full items-center justify-center rounded-2xl bg-slate-50 text-sm text-muted-foreground">
                    当前还没有足够的模型实验结果来汇总特征重要性。
                  </div>
                )}
              </div>

              {featureConsensus.length ? (
                <div className="mt-4 grid gap-3 xl:grid-cols-2">
                  {featureConsensus.slice(0, 4).map((item, index) => (
                    <div key={item.factor} className="rounded-2xl border border-border/50 bg-slate-50 p-3">
                      <div className="flex items-start justify-between gap-3">
                        <div className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Top {index + 1}</div>
                        <div className="rounded-full bg-white px-2 py-1 text-[10px] text-slate-600">
                          {item.supportCount} / {Object.keys(latestModelLab?.models || {}).length} 模型
                        </div>
                      </div>
                      <div className="mt-2">
                        {item.formula || item.score !== undefined ? (
                          <button
                            onClick={() => setSelectedRunId(item.factor)}
                            className="break-all text-left font-mono text-xs font-medium text-sky-700 underline decoration-sky-400 underline-offset-4 hover:text-sky-900"
                          >
                            {item.factor}
                          </button>
                        ) : (
                          <div className="break-all font-mono text-xs font-medium text-foreground">{item.factor}</div>
                        )}
                      </div>
                      <div className="mt-2 text-[11px] leading-5 text-slate-600">
                        模型结论: {item.models.join(' / ')}
                      </div>
                      <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-slate-600">
                        <div>共识分 {formatNumber(item.consensusImportance, 2)}</div>
                        <div>均值权重 {formatNumber(item.avgImportance, 2)}</div>
                        <div>主导轨道 {item.dominantTrackLabel}</div>
                        <div>轨道数 {item.familyFeatureCount}</div>
                        <div>Score {item.score !== undefined ? formatNumber(item.score, 2) : '--'}</div>
                        <div>IC {item.ic !== undefined ? formatNumber(item.ic, 3) : '--'}</div>
                      </div>
                      {item.formula ? (
                        <div className="mt-2 break-words text-[11px] leading-5 text-slate-500">
                          {truncate(item.formula, 100)}
                        </div>
                      ) : null}
                    </div>
                  ))}
                </div>
              ) : null}
            </div>

            {/* ── Cell 2: 特征重要性 / 预测摘要 (top-right) ───────── */}
            <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
              <div className="mb-3 flex items-center gap-2 text-sm font-medium text-foreground">
                <Sparkles className="h-4 w-4 text-emerald-500" />
                特征重要性 / 预测摘要
              </div>
              <div className="space-y-4">
                {Object.entries(latestModelLab?.models || {}).map(([modelName, payload]) => (
                  <div key={modelName} className="rounded-2xl bg-slate-50 p-3">
                    <div className="flex min-w-0 items-center justify-between gap-3">
                      <div className="min-w-0 truncate font-medium text-foreground">{modelName}</div>
                      <div className="shrink-0 text-xs text-muted-foreground">Net PnL {formatNumber(payload.total_pnl, 3)}</div>
                    </div>
                    <div className="mt-2 grid gap-2 text-[11px] text-slate-600 sm:grid-cols-2 2xl:grid-cols-4">
                      <div>IC {formatNumber(payload.avg_daily_ic, 4)}</div>
                      <div>RankIC {formatNumber(payload.avg_daily_rank_ic, 4)}</div>
                      <div>Sharpe {formatNumber(payload.avg_sharpe, 2)}</div>
                      <div>TVR {formatNumber(payload.avg_turnover ?? 0, 3)}</div>
                      <div>Fee {formatNumber(payload.total_fee ?? 0, 4)}</div>
                      <div>Hit {formatPercent(payload.hit_ratio * 100)}</div>
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {payload.top_features.slice(0, 5).map((item) => (
                        <span key={`${modelName}-${item.factor}`} className="rounded-full bg-white px-3 py-1 text-[11px] text-slate-700">
                          {truncate(item.factor, 18)} · {formatNumber(item.importance, 2)}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
                {!(latestModelLab?.models && Object.keys(latestModelLab.models).length) ? (
                  <div className="text-sm text-muted-foreground">当前还没有模型实验结果；现在只要有 1 个有效因子，也可以产出模型实验和整体因子输出。</div>
                ) : null}
              </div>
            </div>

            {/* ── Cell 3: 入模因子清单 (bottom-left, half-height) ──── */}
            <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
              <div className="mb-3 text-sm font-medium text-foreground">入模因子清单</div>
              <div className="max-h-[260px] overflow-y-auto space-y-2 pr-1">
                {(latestModelLab?.selected_factors || []).map((factor) => (
                  <div key={factor.run_id} className="rounded-2xl bg-slate-50 p-3">
                    <div className="flex min-w-0 items-center justify-between gap-3">
                      <button
                        onClick={() => setSelectedRunId(factor.run_id)}
                        className="min-w-0 truncate text-left font-mono text-xs font-medium text-sky-700 underline decoration-sky-400 underline-offset-4 hover:text-sky-900"
                      >
                        {factor.run_id}
                      </button>
                      <div className="shrink-0 text-xs text-muted-foreground">Score {formatNumber(factor.score, 2)}</div>
                    </div>
                    <div className="mt-2 break-words text-xs leading-6 text-slate-700">{truncate(factor.formula, 120)}</div>
                  </div>
                ))}
                {!(latestModelLab?.selected_factors || []).length ? (
                  <div className="text-sm text-muted-foreground">还没有滚动实验结果。</div>
                ) : null}
              </div>
            </div>

            {/* ── Cell 4: 整体因子输出 (bottom-right) ──────────────── */}
            <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
              <div className="mb-1 text-sm font-medium text-foreground">整体因子输出</div>
              <div className="mb-3 text-xs leading-5 text-muted-foreground">
                Rolling Model Lab 每次只保留最佳模型的合成 pq 文件，可直接按提交格式使用。点击卡片可查看详情、因子相关性并填入回测结果。
              </div>
              {latestModelLab?.ensemble_outputs && Object.keys(latestModelLab.ensemble_outputs).length ? (
                <div className="grid gap-3 sm:grid-cols-2">
                  {Object.entries(latestModelLab.ensemble_outputs).map(([mName, path]) => (
                    <button
                      key={mName}
                      onClick={() => setEnsembleModalModel(mName)}
                      className="rounded-2xl bg-slate-50 p-3 text-left transition-colors hover:bg-slate-100"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">{mName}</div>
                        {mName === latestModelLab?.best_model && (
                          <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-medium text-emerald-700">BEST</span>
                        )}
                      </div>
                      <div className="mt-2 break-all font-mono text-[11px] leading-5 text-slate-700">{String(path).split('/').pop()}</div>
                      <div className="mt-1 text-[10px] text-muted-foreground">点击查看详情 →</div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-muted-foreground">整体因子输出将在首次 Model Lab 运行后出现（每积累 10 个有效因子触发一次）。</div>
              )}
            </div>
          </div>
        </Panel>

      </div>

      {selectedRunId ? <ResearchModal runId={selectedRunId} onClose={() => setSelectedRunId(null)} /> : null}
      {ensembleModalModel && latestModelLab ? (
        <EnsembleModal
          modelName={ensembleModalModel}
          modelLab={latestModelLab}
          factorCorrelations={factorCorrelations}
          onOpenFactor={setSelectedRunId}
          onClose={() => setEnsembleModalModel(null)}
        />
      ) : null}
    </div>
  );
};
