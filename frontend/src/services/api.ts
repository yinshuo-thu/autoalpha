import type {
  ApiResponse,
  ExecutionProgress,
  Factor,
  LogEntry,
  RealtimeMetrics,
  Task,
  WsMessage,
} from '@/types';

const BASE = '';

function makeApiResponse<T>(payload: unknown): ApiResponse<T> {
  if (
    payload &&
    typeof payload === 'object' &&
    'success' in (payload as Record<string, unknown>)
  ) {
    return payload as ApiResponse<T>;
  }
  return { success: true, data: payload as T };
}

async function request<T = any>(path: string, options: RequestInit = {}): Promise<ApiResponse<T>> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers ?? {}) },
    ...options,
  });
  const text = await res.text();
  let parsed: unknown = null;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
  }
  if (!res.ok) {
    const message =
      typeof parsed === 'object' && parsed && 'error' in (parsed as Record<string, unknown>)
        ? String((parsed as Record<string, unknown>).error)
        : `API Error ${res.status}`;
    throw new Error(message);
  }
  return makeApiResponse<T>(parsed);
}

function emptyMetrics(): RealtimeMetrics {
  return {
    ic: 0,
    icir: 0,
    rankIc: 0,
    rankIcir: 0,
    score: 0,
    turnover: 0,
    annualReturn: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    totalFactors: 0,
    highQualityFactors: 0,
    mediumQualityFactors: 0,
    lowQualityFactors: 0,
    passGatesCount: 0,
    submissionReadyCount: 0,
    top10Factors: [],
  };
}

function buildTopFactors(rawFactors: any[]): NonNullable<RealtimeMetrics['top10Factors']> {
  return rawFactors
    .slice()
    .sort((a: any, b: any) => {
      const aReady = Number(Boolean(a.submissionReadyFlag ?? a.submission_ready_flag ?? a.submission_ready_flag));
      const bReady = Number(Boolean(b.submissionReadyFlag ?? b.submission_ready_flag ?? b.submission_ready_flag));
      const aPass = Number(Boolean(a.passGates ?? a.PassGates));
      const bPass = Number(Boolean(b.passGates ?? b.PassGates));
      const aScore = Number(a.score ?? a.Score ?? a.backtestResults?.Score ?? 0);
      const bScore = Number(b.score ?? b.Score ?? b.backtestResults?.Score ?? 0);
      const aRankIc = Number(a.rankIc ?? a.rank_ic ?? 0);
      const bRankIc = Number(b.rankIc ?? b.rank_ic ?? 0);
      return bReady - aReady || bPass - aPass || bScore - aScore || bRankIc - aRankIc;
    })
    .slice(0, 10)
    .map((factor: any) => ({
      factorName: factor.factorName || factor.factor_name || 'Unknown',
      factorExpression: factor.factorExpression || factor.factor_expression || factor.formula || '',
      factorDescription: factor.factorDescription || factor.factor_description || factor.recommendation || '',
      score: factor.score ?? factor.Score ?? factor.backtestResults?.Score ?? 0,
      turnover: factor.turnover ?? factor.Turnover ?? factor.backtestResults?.Turnover ?? 0,
      passGates: Boolean(factor.passGates ?? factor.PassGates ?? factor.backtestResults?.['Pass Gates']),
      submissionReadyFlag: Boolean(
        factor.submissionReadyFlag ??
        factor.submission_ready_flag ??
        factor.backtestResults?.['Submission Ready']
      ),
      classification: factor.classification,
      recommendation: factor.recommendation,
      reason: factor.reason,
      submissionPath: factor.submissionPath ?? factor.submission_path,
      submissionDir: factor.submissionDir ?? factor.submission_dir,
      metadataPath: factor.metadataPath ?? factor.metadata_path,
      sanityReport: factor.sanityReport ?? factor.sanity_report,
      gatesDetail: factor.gatesDetail ?? factor.gates_detail,
      rankIc: factor.rankIc ?? factor.rank_ic ?? 0,
      rankIcir: factor.rankIcir ?? factor.rank_ic_ir ?? factor.icir ?? 0,
      ic: factor.ic ?? factor.IC ?? 0,
      icir: factor.icir ?? factor.IR ?? 0,
      annualReturn: factor.annualReturn ?? factor.score ?? factor.Score ?? 0,
      sharpeRatio: factor.sharpeRatio ?? factor.icir ?? factor.IR ?? 0,
      maxDrawdown: factor.maxDrawdown ?? -((factor.backtestResults?.Turnover ?? factor.turnover ?? factor.Turnover ?? 0) / 1000),
      calmarRatio: factor.calmarRatio ?? 0,
      cumulativeCurve: factor.cumulativeCurve ?? [],
    }));
}

function buildMiningMetrics(rawFactors: any[]): RealtimeMetrics {
  const metrics = emptyMetrics();
  const highs = rawFactors.filter((factor) => factor.quality === 'high');
  const mediums = rawFactors.filter((factor) => factor.quality === 'medium');
  const lows = rawFactors.filter((factor) => factor.quality === 'low');
  const top10Factors = buildTopFactors(rawFactors);
  const best = top10Factors[0];
  const passGatesCount = rawFactors.filter((factor) => factor.passGates || factor.PassGates).length;
  const submissionReadyCount = rawFactors.filter(
    (factor) => factor.submissionReadyFlag || factor.submission_ready_flag
  ).length;
  return {
    ...metrics,
    totalFactors: rawFactors.length,
    highQualityFactors: highs.length,
    mediumQualityFactors: mediums.length,
    lowQualityFactors: lows.length,
    passGatesCount,
    submissionReadyCount,
    top10Factors,
    factorName: best?.factorName,
    score: best?.score ?? 0,
    turnover: best?.turnover ?? 0,
    rankIc: best?.rankIc ?? 0,
    rankIcir: best?.rankIcir ?? 0,
    ic: best?.ic ?? 0,
    icir: best?.icir ?? 0,
    annualReturn: best?.annualReturn ?? 0,
    sharpeRatio: best?.sharpeRatio ?? 0,
    maxDrawdown: best?.maxDrawdown ?? 0,
  };
}

function normalizeLogLevel(message: string): LogEntry['level'] {
  const lower = message.toLowerCase();
  if (lower.includes('fail') || lower.includes('error')) return 'error';
  if (lower.includes('warn')) return 'warning';
  if (lower.includes('pass') || lower.includes('success') || lower.includes('完成')) return 'success';
  return 'info';
}

export interface MiningStartParams {
  direction: string;
  numDirections?: number;
  maxRounds?: number;
  /** 与 maxRounds 相乘为 research_loop --max-iters */
  maxLoops?: number;
  factorsPerHypothesis?: number;
  librarySuffix?: string;
  qualityGateEnabled?: boolean;
  parallelEnabled?: boolean;
}

export async function startMining(params: MiningStartParams): Promise<ApiResponse<{ taskId: string; task: Task }>> {
  const resp = await request<{
    status: string;
    llm_enabled?: boolean;
    max_iters?: number;
    batch_size?: number;
  }>('/api/factory/start', {
    method: 'POST',
    body: JSON.stringify(params),
  });
  if (!resp.success) {
    return { success: false, error: resp.error || 'Failed to start mining' };
  }

  const now = new Date().toISOString();
  const data = resp.data;
  const llmOn = data?.llm_enabled !== false;
  const task: Task = {
    taskId: 'global',
    status: 'running',
    config: {
      userInput: params.direction,
      numDirections: params.numDirections,
      maxRounds: params.maxRounds,
      maxLoops: params.maxLoops,
      librarySuffix: params.librarySuffix,
      qualityGateEnabled: params.qualityGateEnabled,
      parallelExecution: params.parallelEnabled,
    },
    engineMeta: { llmEnabled: llmOn },
    progress: {
      phase: 'planning',
      currentRound: 0,
      totalRounds: params.maxRounds || 3,
      progress: 5,
      message:
        data?.max_iters != null
          ? `已启动：最多 ${data.max_iters} 轮迭代，batch=${data.batch_size ?? '—'}${llmOn ? ' · LLM 开启' : ' · 无 LLM，仅 EA'}`
          : '自动挖掘已启动，等待首批日志...',
      timestamp: now,
    },
    logs: [],
    metrics: emptyMetrics(),
    createdAt: now,
    updatedAt: now,
  };
  return { success: true, data: { taskId: 'global', task }, message: resp.message };
}

function inferMiningPhase(logTail: string): ExecutionProgress['phase'] {
  const s = logTail.toLowerCase();
  if (s.includes('loading datahub') || s.includes('seed prompt')) return 'planning';
  if (s.includes('quick_test') || s.includes('eval]') || s.includes('gates')) return 'backtesting';
  if (s.includes('iteration')) return 'evolving';
  return 'evolving';
}

export async function getMiningStatus(taskId: string): Promise<ApiResponse<{ task: Task }>> {
  const statusResp = await request<{
    global_state: Record<string, unknown> & {
      is_running?: boolean;
      llm_enabled?: boolean;
      research_log_path?: string;
      llm_mining_log_path?: string;
    };
    agents: unknown[];
    cmd_logs?: string[];
    llm_mining_recent?: Record<string, unknown>[];
  }>('/api/factory/status');
  if (!statusResp.success || !statusResp.data) {
    return { success: false, error: statusResp.error || 'Failed to get mining status' };
  }

  const factorsResp = await getFactors({ limit: 500 });
  const factorRows = factorsResp.success && factorsResp.data ? factorsResp.data.factors : [];
  const metrics = buildMiningMetrics(factorRows);
  const running = Boolean(statusResp.data.global_state?.is_running);
  const cmdLogs = statusResp.data.cmd_logs || [];
  const llmMiningRecent = statusResp.data.llm_mining_recent || [];
  const gs = statusResp.data.global_state;
  const logs: LogEntry[] = cmdLogs.map((message, index) => ({
    id: `mining-log-${index}`,
    timestamp: new Date().toISOString(),
    level: normalizeLogLevel(message),
    message,
  }));

  const lastLines = cmdLogs.slice(-12).join('\n');
  const lastOne = cmdLogs.length ? cmdLogs[cmdLogs.length - 1] : '';
  const phase = running ? inferMiningPhase(lastLines) : 'completed';
  const llmEnabled = Boolean(statusResp.data.global_state?.llm_enabled);
  const bestFactorInput = typeof gs?.best_factor === 'string' ? gs.best_factor : '自动挖掘';
  const progressPct = running
    ? Math.min(92, 8 + Math.min(84, cmdLogs.length * 0.35 + metrics.totalFactors * 2))
    : 100;

  const task: Task = {
    taskId,
    status: running ? 'running' : 'completed',
    config: { userInput: bestFactorInput },
    engineMeta: { llmEnabled },
    progress: {
      phase,
      currentRound: 0,
      totalRounds: 1,
      progress: progressPct,
      message: running
        ? lastOne.slice(0, 420) || '引擎运行中（日志见下方，数据来自真实 research_loop 输出）'
        : '自动挖掘已停止',
      timestamp: new Date().toISOString(),
    },
    logs,
    metrics,
    llmMiningRecent,
    logPaths: {
      researchLog: typeof gs?.research_log_path === 'string' ? gs.research_log_path : undefined,
      llmMiningJsonl: typeof gs?.llm_mining_log_path === 'string' ? gs.llm_mining_log_path : undefined,
    },
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  return { success: true, data: { task } };
}

export async function cancelMining(_taskId: string) {
  return request<{ status: string }>('/api/factory/stop', { method: 'POST' });
}

export async function listTasks() {
  return { success: true, data: { tasks: [] } } as ApiResponse<{ tasks: Task[] }>;
}

export interface FactorListParams {
  quality?: string;
  search?: string;
  limit?: number;
  offset?: number;
  library?: string;
}

export interface FactorListResponse {
  factors: Factor[];
  total: number;
  limit: number;
  offset: number;
  metadata?: any;
  libraries?: string[];
}

export async function getFactors(params: FactorListParams = {}): Promise<ApiResponse<FactorListResponse>> {
  const qs = new URLSearchParams();
  if (params.quality) qs.set('quality', params.quality);
  if (params.search) qs.set('search', params.search);
  if (params.limit) qs.set('limit', String(params.limit));
  if (params.offset) qs.set('offset', String(params.offset));
  if (params.library) qs.set('library', params.library);
  return request<FactorListResponse>(`/api/factors?${qs.toString()}`);
}

export async function getFactorDetail(factorId: string) {
  return request<{ factor: any }>(`/api/factors/${factorId}`);
}

export async function listFactorLibraries() {
  return request<{ libraries: string[] }>('/api/factors/libraries');
}

export interface CacheStatusResponse {
  total: number;
  h5_cached: number;
  md5_cached: number;
  need_compute: number;
  factors: Array<{
    factor_id: string;
    factor_name: string;
    status: 'h5_cached' | 'md5_cached' | 'need_compute';
  }>;
}

export interface WarmCacheResponse {
  total: number;
  synced: number;
  skipped: number;
  failed: number;
}

export async function getCacheStatus(library?: string) {
  const qs = new URLSearchParams();
  if (library) qs.set('library', library);
  return request<CacheStatusResponse>(`/api/factors/cache-status?${qs.toString()}`);
}

export async function warmCache(library?: string) {
  const qs = new URLSearchParams();
  if (library) qs.set('library', library);
  return request<WarmCacheResponse>(`/api/factors/warm-cache?${qs.toString()}`, { method: 'POST' });
}

export interface BacktestStartParams {
  factorJson: string;
  factorSource?: string;
  configPath?: string;
}

export async function startBacktest(params: BacktestStartParams) {
  return request<{ taskId: string; task: Task }>('/api/backtest/start', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function getBacktestStatus(taskId: string) {
  return request<{ task: Task }>(`/api/backtest/${taskId}`);
}

export async function cancelBacktest(taskId: string) {
  return request<{ task: Task }>(`/api/backtest/${taskId}`, { method: 'DELETE' });
}

export async function getSystemConfig() {
  return request<{ env: Record<string, string>; factorLibraries: string[]; paths?: Record<string, string> }>('/api/system/config');
}

export async function updateSystemConfig(update: Record<string, string>) {
  return request<{ env: Record<string, string> }>('/api/system/config', {
    method: 'PUT',
    body: JSON.stringify(update),
  });
}

export async function testLlmConnection() {
  return request<any>('/api/system/llm-test', { method: 'POST' });
}

export interface ExecuteFactorParams {
  input: string;
  name?: string;
  postprocess?: 'rank' | 'zscore';
}

export async function executeFactor(params: ExecuteFactorParams) {
  return request<{ factor: any; ideas: any[] }>('/api/formula/execute', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function healthCheck() {
  return request<{ status: string; timestamp: string }>('/api/health');
}

// ── AutoAlpha Loop API ────────────────────────────────────────────────────────

export interface AutoAlphaLoopStatus {
  is_running: boolean;
  total_tested: number;
  total_passing: number;
  best_score: number;
  updated_at: string;
  logs: string[];
}

export interface AutoAlphaKbFactor {
  run_id: string;
  rank?: number;
  formula: string;
  thought_process?: string;
  IC: number;
  IR: number;
  tvr: number;
  Score: number;
  PassGates: boolean;
  status: string;
  generation: number;
  created_at: string;
  eval_days?: number;
  errors?: string;
  parquet_path?: string;
  research_path?: string;
}

export interface AutoAlphaKnowledge extends AutoAlphaLoopStatus {
  pass_rate?: number;
  status_breakdown?: Record<string, number>;
  progress_points?: Array<{
    index: number;
    timestamp: string;
    label: string;
    tested: number;
    passing: number;
    best_score: number;
  }>;
  generation_summary?: Array<{
    generation: number;
    total: number;
    passing: number;
    best_score: number;
  }>;
  factors: AutoAlphaKbFactor[];
  artifacts?: {
    output_files: Array<{
      name: string;
      path: string;
      relative_path: string;
      kind: string;
      size_bytes: number;
      modified_at: string;
    }>;
    research_reports: Array<{
      run_id: string;
      path: string;
      relative_path: string;
      modified_at: string;
      size_bytes: number;
    }>;
  };
}

export interface AutoAlphaBalance {
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

export async function getAutoAlphaLoopStatus() {
  return request<AutoAlphaLoopStatus>('/api/autoalpha/loop/status');
}

export async function startAutoAlphaLoop(params: { rounds: number; ideas: number; days: number }) {
  return request<{ status: string }>('/api/autoalpha/loop/start', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function stopAutoAlphaLoop() {
  return request<{ status: string }>('/api/autoalpha/loop/stop', { method: 'POST' });
}

export async function getAutoAlphaKnowledge() {
  return request<AutoAlphaKnowledge>('/api/autoalpha/knowledge');
}

export async function getAutoAlphaBalance() {
  return request<AutoAlphaBalance>('/api/autoalpha/balance');
}

export async function getAutoAlphaResearch(runId: string) {
  return request<{ report: any }>(`/api/autoalpha/research/${runId}`);
}

export type WsCallback = (msg: WsMessage) => void;

export function connectMiningWs(
  taskId: string,
  onMessage: WsCallback,
  onClose?: () => void,
  onError?: (e: Event) => void
): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/mining/${taskId}`;
  const ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    try {
      onMessage(JSON.parse(event.data) as WsMessage);
    } catch {
      // ignore parse failures
    }
  };
  ws.onclose = () => onClose?.();
  ws.onerror = (event) => onError?.(event);
  return ws;
}
