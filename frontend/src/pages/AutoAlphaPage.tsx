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
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { BrainCircuit, FileStack, FlaskConical, FolderSync, Sparkles, Wand2 } from 'lucide-react';
import { Progress } from '@/components/ui/Progress';

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
  parent_run_ids?: string[];
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
  max_drawdown: number;
  hit_ratio: number;
  cumulative_curve: ModelLabPoint[];
  daily_pnl_curve: ModelLabPoint[];
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

interface InspirationRecord {
  id: number;
  kind: string;
  title: string;
  source: string;
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

const COLORS = {
  used: '#f97316',
  remaining: '#10b981',
  tested: '#60a5fa',
  passing: '#34d399',
  passRate: '#8b5cf6',
  bestScore: '#0f766e',
  score: '#15803d',
  ic: '#2563eb',
  histogram: '#7c3aed',
  error: '#ef4444',
  pnlA: '#0f766e',
  pnlB: '#2563eb',
  pnlC: '#9333ea',
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

function buildMergedModelCurves(models: Record<string, ModelLabModelSummary> | undefined) {
  const merged = new Map<string, Record<string, string | number>>();
  Object.entries(models || {}).forEach(([modelName, payload]) => {
    payload.cumulative_curve.forEach((point) => {
      const row = merged.get(point.date) || { date: point.date, label: point.date.slice(5) };
      row[modelName] = point.value;
      merged.set(point.date, row);
    });
  });
  return Array.from(merged.values()).sort((a, b) => String(a.date).localeCompare(String(b.date)));
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

export const AutoAlphaPage: React.FC = () => {
  const [status, setStatus] = useState<LoopStatus | null>(null);
  const [knowledge, setKnowledge] = useState<KnowledgePayload | null>(null);
  const [balance, setBalance] = useState<BalancePayload | null>(null);
  const [modelLab, setModelLab] = useState<ModelLabPayload | null>(null);
  const [inspirations, setInspirations] = useState<InspirationPayload | null>(null);
  const [rounds, setRounds] = useState(10);
  const [ideas, setIdeas] = useState(3);
  const [days, setDays] = useState(0);
  const [targetValid, setTargetValid] = useState(100);
  const [promptTitle, setPromptTitle] = useState('');
  const [promptInput, setPromptInput] = useState('');
  const [promptBusy, setPromptBusy] = useState(false);
  const [manualBusy, setManualBusy] = useState(false);
  const [manualFactor, setManualFactor] = useState<ManualFactor | null>(null);
  const [pageMessage, setPageMessage] = useState('');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let mounted = true;

    const loadAll = async () => {
      const [statusResult, knowledgeResult, balanceResult, modelLabResult, inspirationResult] = await Promise.allSettled([
        fetchJson<LoopStatus>('/api/autoalpha/loop/status'),
        fetchJson<KnowledgePayload>('/api/autoalpha/knowledge'),
        fetchJson<BalancePayload>('/api/autoalpha/balance'),
        fetchJson<ModelLabPayload>('/api/autoalpha/model-lab'),
        fetchJson<InspirationPayload>('/api/autoalpha/inspirations'),
      ]);

      if (!mounted) return;
      if (statusResult.status === 'fulfilled') setStatus(statusResult.value);
      if (knowledgeResult.status === 'fulfilled') setKnowledge(knowledgeResult.value);
      if (balanceResult.status === 'fulfilled') setBalance(balanceResult.value);
      if (modelLabResult.status === 'fulfilled') setModelLab(modelLabResult.value);
      if (inspirationResult.status === 'fulfilled') setInspirations(inspirationResult.value);
    };

    loadAll();
    pollRef.current = setInterval(loadAll, 4000);

    fetchJson<RuntimeConfigPayload>('/api/system/config')
      .then((cfg) => {
        const env = cfg.env || {};
        setRounds(Number(env.AUTOALPHA_DEFAULT_ROUNDS || 0));
        setIdeas(Number(env.AUTOALPHA_DEFAULT_IDEAS || 4));
        setDays(Number(env.AUTOALPHA_DEFAULT_DAYS || 0));
        setTargetValid(Number(env.AUTOALPHA_DEFAULT_TARGET_VALID || 100));
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
    () =>
      progressPoints.map((point) => ({
        ...point,
        pass_rate: point.tested > 0 ? Number(((point.passing / point.tested) * 100).toFixed(2)) : 0,
      })),
    [progressPoints]
  );
  const latestModelLab = modelLab?.latest ?? null;
  const modelLabCurve = buildMergedModelCurves(latestModelLab?.models);

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
      setPageMessage('灵感已写入 AutoAlpha 灵感库，并会自动进入后续挖掘上下文。');
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

  const balanceDonutData = [
    { name: '已用', value: balance?.used ?? 0, fill: COLORS.used },
    { name: '剩余', value: Math.max(balance?.remaining ?? 0, 0), fill: COLORS.remaining },
  ];

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
          <StatCard label="Prompt 灵感" value={String(inspirations?.count ?? 0)} helper="Prompt / URL / 目录文件" accent="bg-violet-50" />
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

      <Panel
        title="额度包与成本"
        subtitle="按真实额度包口径展示，因子成本单独成行，避免数字拥挤。"
        right={<div className={`rounded-full bg-emerald-500/10 px-4 py-2 text-sm font-medium ${quotaTone(balance?.quota_status ?? 'healthy')}`}>{balance?.quota_status ?? 'healthy'}</div>}
        className="overflow-hidden"
      >
        <div className="grid min-w-0 gap-5 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
          <div className="min-w-0 overflow-hidden rounded-[28px] bg-[linear-gradient(135deg,#0b3d30_0%,#0f6b53_58%,#138163_100%)] p-5 text-white shadow-[0_22px_70px_rgba(15,118,110,0.18)] md:p-6">
            <div className="inline-flex items-center rounded-full border border-white/20 bg-white/12 px-4 py-2 text-sm font-medium text-emerald-50">
              <Sparkles className="mr-2 h-4 w-4" />
              API Credit
            </div>
            <div className="mt-6 text-4xl font-semibold leading-none tracking-tight text-white md:text-5xl">额度包</div>
            <div className="mt-6 min-w-0 break-words text-[clamp(2rem,5vw,4.5rem)] font-semibold leading-none text-white">
              {formatMoney(balance?.remaining ?? 0)}
              <span className="ml-2 text-[clamp(1.1rem,2vw,1.8rem)] text-emerald-50/90">/ {formatMoney(balance?.total_quota ?? 0)}</span>
            </div>
            <div className="mt-5 rounded-3xl bg-white/12 p-4">
              <div className="mb-2 flex items-center justify-between text-sm text-emerald-50">
                <span>剩余额度</span>
                <span>{formatPercent(balance?.remaining_pct ?? 0)}</span>
              </div>
              <div className="h-4 overflow-hidden rounded-full bg-white/20">
                <div className="h-full rounded-full bg-emerald-300" style={{ width: `${balance?.remaining_pct ?? 0}%` }} />
              </div>
            </div>
          </div>

          <div className="grid min-w-0 gap-4">
            <div className="grid min-w-0 gap-4 md:grid-cols-3">
              <StatCard label="总额度" value={formatMoney(balance?.total_quota ?? 0)} helper="真实额度包口径" valueClassName="whitespace-nowrap text-[clamp(1.5rem,2vw,2.35rem)] tabular-nums" />
              <StatCard label="已用额度" value={formatMoney(balance?.used ?? 0)} helper={`${formatPercent(balance?.used_pct ?? 0)} 已使用`} accent="bg-orange-50" valueClassName="whitespace-nowrap text-[clamp(1.5rem,2vw,2.35rem)] tabular-nums" />
              <StatCard label="剩余额度" value={formatMoney(balance?.remaining ?? 0)} helper={`${formatPercent(balance?.remaining_pct ?? 0)} 剩余`} accent="bg-emerald-50" valueClassName="whitespace-nowrap text-[clamp(1.5rem,2vw,2.35rem)] tabular-nums" />
            </div>

            <div className="grid min-w-0 gap-3">
              <StatCard label="因子成本" value={formatMoney(balance?.avg_cost_per_factor ?? 0)} helper={`${formatInteger(balance?.avg_tokens_per_factor ?? 0)} tokens / 因子`} accent="bg-sky-50" valueClassName="whitespace-nowrap text-[clamp(1.35rem,1.8vw,2rem)] tabular-nums" />
              <StatCard label="有效因子成本" value={formatMoney(balance?.avg_cost_per_valid_factor ?? 0)} helper={`${formatInteger(balance?.avg_tokens_per_valid_factor ?? 0)} tokens / 通过因子`} accent="bg-violet-50" valueClassName="whitespace-nowrap text-[clamp(1.35rem,1.8vw,2rem)] tabular-nums" />
            </div>

            <div className="grid min-w-0 gap-4 lg:grid-cols-[minmax(0,0.72fr)_minmax(0,1.28fr)]">
              <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
                <div className="h-52">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={balanceDonutData} dataKey="value" nameKey="name" innerRadius={52} outerRadius={82} paddingAngle={4} stroke="none">
                        {balanceDonutData.map((item) => (
                          <Cell key={item.name} fill={item.fill} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value: number) => formatMoney(Number(value || 0))} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="-mt-2 text-center">
                  <div className="text-xs uppercase tracking-[0.22em] text-muted-foreground">额度占用</div>
                  <div className="mt-2 text-4xl font-semibold text-foreground">{formatPercent(balance?.used_pct ?? 0)}</div>
                </div>
              </div>

              <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
                <div className="mb-2 flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">使用百分比</span>
                  <span className="font-medium text-foreground">{formatPercent(balance?.used_pct ?? 0)}</span>
                </div>
                <Progress value={balance?.used_pct ?? 0} />
                <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
                  <span>已用 {formatPercent(balance?.used_pct ?? 0)}</span>
                  <span>剩余 {formatPercent(balance?.remaining_pct ?? 0)}</span>
                </div>
                <div className="mt-5 grid gap-3 md:grid-cols-3">
                  <div className="rounded-2xl bg-slate-50 p-3">
                    <div className="text-xs text-muted-foreground">实际调用</div>
                    <div className="mt-2 text-xl font-semibold text-foreground">{formatInteger(balance?.total_factors ?? 0)} 次</div>
                  </div>
                  <div className="rounded-2xl bg-slate-50 p-3">
                    <div className="text-xs text-muted-foreground">通过率</div>
                    <div className="mt-2 text-xl font-semibold text-foreground">{formatPercent(balance?.pass_rate ?? 0)}</div>
                  </div>
                  <div className="rounded-2xl bg-slate-50 p-3">
                    <div className="text-xs text-muted-foreground">Token 消耗</div>
                    <div className="mt-2 text-xl font-semibold text-foreground">{formatInteger(balance?.est_total_tokens ?? 0)}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Panel>

      <div className="grid min-w-0 gap-6 xl:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)]">
        <Panel
          title="Prompt Lab"
          subtitle="现在可以直接在 AutoAlpha 页面输入文字链接或纯提示词。内容会落到 AutoAlpha 目录里的灵感库，并自动进入后续因子挖掘上下文。"
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
                      placeholder="输入研究灵感、文章链接、市场观察，或直接输入一段 DSL。保存后会进入 AutoAlpha 灵感库。"
                      className="mt-2 h-40 w-full rounded-2xl border border-border/60 bg-slate-50 px-4 py-3 text-sm outline-none"
                    />
                  </label>
                </div>
                <div className="mt-4 flex flex-wrap gap-3">
                  <button onClick={handleAddInspiration} disabled={promptBusy} className="rounded-2xl bg-slate-950 px-4 py-3 text-sm text-white transition-colors hover:bg-slate-800 disabled:opacity-50">
                    {promptBusy ? '处理中...' : '加入灵感库'}
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
                          <div className="truncate text-xs font-medium text-foreground leading-tight">{item.title}</div>
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

        <Panel title="循环控制与实时日志" subtitle="直接启动全量数据挖掘；rounds=0 表示持续运行，直到达到目标有效因子数或手动停止。">
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

      <div className="grid min-w-0 gap-6 xl:grid-cols-[minmax(0,1.08fr)_minmax(0,0.92fr)]">
        <Panel title="Rolling Model Lab" subtitle="基于当前已获得的有效因子做滚动训练实验：支持 1 个或更多有效因子，半年训练、半年测试、继续向后滚动，并输出线性模型 / LightGBM 的预测效果、PnL 曲线和整体因子 parquet。">
          <div className="grid min-w-0 gap-4 md:grid-cols-2 2xl:grid-cols-4">
            <StatCard label="Run ID" value={latestModelLab?.run_id || '--'} helper={latestModelLab ? formatDateTime(latestModelLab.created_at) : '还没有实验'} valueClassName="text-lg" />
            <StatCard label="选中因子数" value={String(latestModelLab?.selected_factor_count ?? 0)} helper={`目标 ${latestModelLab?.target_valid_count ?? 0}`} accent="bg-violet-50" />
            <StatCard label="滚动窗口数" value={String(latestModelLab?.window_count ?? 0)} helper={latestModelLab ? `${latestModelLab.train_days}/${latestModelLab.test_days}/${latestModelLab.step_days}` : 'train/test/step'} accent="bg-sky-50" />
            <StatCard label="最优模型" value={latestModelLab?.best_model || '--'} helper="线性 / LGB 竞赛" accent="bg-emerald-50" />
          </div>

          <div className="mt-5 grid min-w-0 gap-5 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)]">
            <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
              <div className="mb-3 text-sm font-medium text-foreground">累计 PnL 曲线</div>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={modelLabCurve}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="label" tick={{ fill: '#64748b', fontSize: 11 }} />
                    <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
                    <Tooltip />
                    {Object.keys(latestModelLab?.models || {}).map((modelName, index) => (
                      <Line
                        key={modelName}
                        type="monotone"
                        dataKey={modelName}
                        stroke={[COLORS.pnlA, COLORS.pnlB, COLORS.pnlC][index % 3]}
                        strokeWidth={3}
                        dot={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
              <div className="mb-3 text-sm font-medium text-foreground">模型对比</div>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={modelComparison}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="model" tick={{ fill: '#64748b', fontSize: 11 }} />
                    <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="ic" name="Avg IC" fill={COLORS.ic} radius={[8, 8, 0, 0]} />
                    <Bar dataKey="rankIc" name="Avg Rank IC" fill={COLORS.histogram} radius={[8, 8, 0, 0]} />
                    <Bar dataKey="sharpe" name="Sharpe" fill={COLORS.passing} radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="mt-5 grid min-w-0 gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
              <div className="mb-3 flex items-center gap-2 text-sm font-medium text-foreground">
                <FileStack className="h-4 w-4 text-sky-500" />
                入模因子清单
              </div>
              <div className="space-y-3">
                {(latestModelLab?.selected_factors || []).slice(0, 8).map((factor) => (
                  <div key={factor.run_id} className="rounded-2xl bg-slate-50 p-3">
                    <div className="flex min-w-0 items-center justify-between gap-3">
                      <div className="min-w-0 truncate font-mono text-xs text-foreground">{factor.run_id}</div>
                      <div className="shrink-0 text-xs text-muted-foreground">Score {formatNumber(factor.score, 2)}</div>
                    </div>
                    <div className="mt-2 break-words text-xs leading-6 text-slate-700">{truncate(factor.formula, 120)}</div>
                  </div>
                ))}
                {!(latestModelLab?.selected_factors || []).length ? <div className="text-sm text-muted-foreground">还没有滚动实验结果。</div> : null}
              </div>
            </div>

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
                      <div className="shrink-0 text-xs text-muted-foreground">PnL {formatNumber(payload.total_pnl, 3)}</div>
                    </div>
                    <div className="mt-2 grid gap-2 text-[11px] text-slate-600 sm:grid-cols-2 2xl:grid-cols-4">
                      <div>IC {formatNumber(payload.avg_daily_ic, 4)}</div>
                      <div>RankIC {formatNumber(payload.avg_daily_rank_ic, 4)}</div>
                      <div>Sharpe {formatNumber(payload.avg_sharpe, 2)}</div>
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
                {!(latestModelLab?.models && Object.keys(latestModelLab.models).length) ? <div className="text-sm text-muted-foreground">当前还没有模型实验结果；现在只要有 1 个有效因子，也可以产出模型实验和整体因子输出。</div> : null}
              </div>
            </div>
          </div>
          {latestModelLab?.ensemble_outputs && Object.keys(latestModelLab.ensemble_outputs).length ? (
            <div className="mt-5 min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
              <div className="mb-3 text-sm font-medium text-foreground">整体因子输出</div>
              <div className="grid min-w-0 gap-3 md:grid-cols-2">
                {Object.entries(latestModelLab.ensemble_outputs).map(([modelName, path]) => (
                  <div key={modelName} className="rounded-2xl bg-slate-50 p-3">
                    <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">{modelName}</div>
                    <div className="mt-2 break-all font-mono text-xs leading-6 text-foreground">{path}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </Panel>

        <Panel title="研究进程综述">
          <div className="grid min-w-0 gap-5">
            <div className="min-w-0 rounded-3xl border border-border/50 bg-white/90 p-4">
              <div className="mb-3 text-sm font-medium text-foreground">测试进度与通过数</div>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={progressChartPoints}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="timestamp" tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={formatAxisTime} minTickGap={28} />
                    <YAxis yAxisId="count" tick={{ fill: '#64748b', fontSize: 11 }} />
                    <YAxis
                      yAxisId="rate"
                      orientation="right"
                      domain={[0, 10]}
                      tick={{ fill: '#92400e', fontSize: 11 }}
                      tickFormatter={(value) => `${Number(value).toFixed(0)}%`}
                    />
                    <Tooltip formatter={(value: number, name: string) => (name === '通过比例' ? `${formatNumber(Number(value), 2)}%` : formatNumber(Number(value), 0))} />
                    <Legend />
                    <Area yAxisId="count" type="monotone" dataKey="tested" name="tested" stroke={COLORS.tested} fill={COLORS.tested} fillOpacity={0.15} />
                    <Area yAxisId="count" type="monotone" dataKey="passing" name="passing" stroke={COLORS.passing} fill={COLORS.passing} fillOpacity={0.15} />
                    <Line
                      yAxisId="rate"
                      type="monotone"
                      dataKey="pass_rate"
                      name="通过比例"
                      stroke={COLORS.passRate}
                      strokeWidth={2}
                      strokeDasharray="6 5"
                      dot={false}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
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
      </div>

    </div>
  );
};
