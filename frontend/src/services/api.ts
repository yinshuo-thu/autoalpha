/**
 * QuantaAlpha API Service
 *
 * Centralized API client for communicating with the FastAPI backend.
 * Uses fetch (no extra dependency) with the Vite proxy (/api -> localhost:8000).
 */

import type {
  ApiResponse,
  Factor,
  Task,
  WsMessage,
} from '@/types';

// ========================== HTTP Helpers ==========================

const BASE = ''; // Vite proxy handles /api -> backend

async function request<T = any>(
  path: string,
  options: RequestInit = {}
): Promise<ApiResponse<T>> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers as any },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API Error ${res.status}: ${text}`);
  }
  return res.json();
}

// ========================== Mining API ==========================

export interface MiningStartParams {
  direction: string;
  numDirections?: number;
  maxRounds?: number;
  maxLoops?: number;
  factorsPerHypothesis?: number;
  librarySuffix?: string;
  qualityGateEnabled?: boolean;
  parallelEnabled?: boolean;
}

export async function startMining(params: MiningStartParams) {
  await request('/api/factory/start', { method: 'POST' });
  return { taskId: 'global', task: { id: 'global', status: 'running', progress: 0, logs: [], timestamp: Date.now() } as Task };
}

export async function getMiningStatus(taskId: string) {
  const res = await request<{global_state: any, agents: any[]}>('/api/factory/status');
  const running = res.global_state.is_running;
  const recentMsg = Array.isArray(res.agents) ? res.agents.map(a => `[${a.name}] ${a.status} - ${a.task}`) : [];
  return { task: { id: 'global', status: running ? 'running' : 'idle', progress: running ? (res.global_state.total_factors > 0 ? Math.min(99, res.global_state.total_factors) : 10) : 100, logs: recentMsg, timestamp: Date.now(), metrics: null } as Task };
}

export async function cancelMining(taskId: string) {
  return request('/api/factory/stop', { method: 'POST' });
}

export async function listTasks() {
  return { tasks: [] };
}

// ========================== Factor API ==========================

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

export async function getFactors(params: FactorListParams = {}) {
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

// ========================== Factor Cache API ==========================

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
  return request<WarmCacheResponse>(`/api/factors/warm-cache?${qs.toString()}`, {
    method: 'POST',
  });
}

// ========================== Backtest API ==========================

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
  return request(`/api/backtest/${taskId}`, { method: 'DELETE' });
}

// ========================== System Config API ==========================

export async function getSystemConfig() {
  return request<{ env: Record<string, string>; experimentYaml: string; factorLibraries: string[] }>(
    '/api/system/config'
  );
}

export async function updateSystemConfig(update: Record<string, string>) {
  return request('/api/system/config', {
    method: 'PUT',
    body: JSON.stringify(update),
  });
}

// ========================== Health Check ==========================

export async function healthCheck() {
  return request<{ status: string; timestamp: string }>('/api/health');
}

// ========================== WebSocket ==========================

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

  ws.onopen = () => {
    console.log(`[WS] Connected to ${taskId}`);
  };

  ws.onmessage = (event) => {
    try {
      const msg: WsMessage = JSON.parse(event.data);
      onMessage(msg);
    } catch (e) {
      console.warn('[WS] Failed to parse message:', event.data);
    }
  };

  ws.onclose = () => {
    console.log(`[WS] Disconnected from ${taskId}`);
    onClose?.();
  };

  ws.onerror = (e) => {
    console.error('[WS] Error:', e);
    onError?.(e);
  };

  // Heartbeat every 30s
  const heartbeat = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send('ping');
    } else {
      clearInterval(heartbeat);
    }
  }, 30000);

  return ws;
}
