/**
 * TaskContext — Global Task State Management
 *
 * Lifts mining and backtest task state, WebSocket connection, and polling logic
 * to App level, so running state is not lost when switching pages.
 */

import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';
import type {
  Task,
  TaskConfig,
  LogEntry,
  TimeSeriesData,
} from '@/types';
import {
  startMining as apiStartMining,
  getMiningStatus,
  cancelMining as apiCancelMining,
  startBacktest as apiStartBacktest,
  getBacktestStatus,
  cancelBacktest as apiCancelBacktest,
  healthCheck,
} from '@/services/api';
import type { BacktestStartParams } from '@/services/api';
import { getDefaultMiningDirection } from '@/utils/miningDirections';
import { getAutoAlphaConfigRaw } from '@/utils/autoalphaStorage';

// ========================== Backtest local type ==========================

export interface BacktestTask {
  taskId: string;
  status: string;
  progress: {
    phase: string;
    progress: number;
    message: string;
    timestamp: string;
  };
  logs: LogEntry[];
  metrics: Record<string, any>;
  config: Record<string, any>;
  createdAt: string;
  updatedAt: string;
}

// ========================== Context Value ==========================

interface TaskContextValue {
  // Backend health
  backendAvailable: boolean | null;

  // ---- Mining ----
  miningTask: Task | null;
  miningEquityCurve: TimeSeriesData[];
  miningDrawdownCurve: TimeSeriesData[];
  miningIcTimeSeries: TimeSeriesData[];
  startMining: (config: TaskConfig) => void;
  stopMining: () => void;
  resetMiningTask: () => void;

  // ---- Backtest ----
  backtestTask: BacktestTask | null;
  backtestLogs: LogEntry[];
  startBacktestTask: (params: BacktestStartParams) => Promise<void>;
  stopBacktestTask: () => void;
}

const TaskContext = createContext<TaskContextValue | null>(null);

// ========================== Provider ==========================

export const TaskProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // ---- Backend health ----
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);

  const probeBackendHealth = useCallback(async () => {
    try {
      await healthCheck();
      setBackendAvailable(true);
      return true;
    } catch {
      setBackendAvailable(false);
      return false;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    const runProbe = async () => {
      try {
        await healthCheck();
        if (!cancelled) {
          setBackendAvailable(true);
        }
      } catch {
        if (!cancelled) {
          setBackendAvailable(false);
        }
      }
    };

    runProbe();
    const timer = setInterval(runProbe, 5000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  // ==================================================================
  // MINING
  // ==================================================================
  const [miningTask, setMiningTask] = useState<Task | null>(null);
  const [miningEquityCurve, setMiningEquityCurve] = useState<TimeSeriesData[]>([]);
  const [miningDrawdownCurve, setMiningDrawdownCurve] = useState<TimeSeriesData[]>([]);
  const [miningIcTimeSeries, setMiningIcTimeSeries] = useState<TimeSeriesData[]>([]);

  const miningPollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Start mining (real backend)
  const startRealMining = useCallback(
    async (config: TaskConfig) => {
      try {
        // Load defaults from localStorage
        let defaults: any = {};
        const savedConfig = getAutoAlphaConfigRaw();
        if (savedConfig) {
          try {
            defaults = JSON.parse(savedConfig);
          } catch {}
        }

        const direction =
          config.useCustomMiningDirection
            ? (getDefaultMiningDirection() || '价量因子挖掘')
            : (config.userInput && config.userInput.trim()) || getDefaultMiningDirection() || '价量因子挖掘';
        const resp = await apiStartMining({
          direction,
          numDirections: config.numDirections || defaults.defaultNumDirections || 2,
          maxRounds: config.maxRounds || defaults.defaultMaxRounds || 3,
          maxLoops:
            config.maxLoops ??
            (typeof defaults.defaultMaxLoops === 'number' ? defaults.defaultMaxLoops : undefined) ??
            defaults.defaultMaxRounds ??
            5,
          librarySuffix: config.librarySuffix || defaults.defaultLibrarySuffix || undefined,
          qualityGateEnabled: config.qualityGateEnabled ?? defaults.qualityGateEnabled ?? true,
          parallelEnabled: config.parallelExecution ?? defaults.parallelExecution ?? false,
        });
        if (!resp.success || !resp.data) throw new Error(resp.error || 'Failed');

        const taskData = resp.data.task as Task;
        // Initialize metrics with empty top10Factors to avoid stale data
        if (taskData.metrics) {
            taskData.metrics.top10Factors = [];
            taskData.metrics.totalFactors = 0;
            taskData.metrics.highQualityFactors = 0;
            taskData.metrics.mediumQualityFactors = 0;
            taskData.metrics.lowQualityFactors = 0;
        }
        setMiningTask(taskData);
        setMiningEquityCurve([]);
        setMiningDrawdownCurve([]);
        setMiningIcTimeSeries([]);

        // Polling: /api/factory/status + leaderboard；进度与日志均为真实引擎输出（不再注入随机曲线）
        miningPollingRef.current = setInterval(async () => {
          try {
            const r = await getMiningStatus(resp.data!.taskId);
            if (r.data?.task) {
              const t = r.data.task as Task;
              setMiningTask(t);
              if (t.status === 'completed' || t.status === 'failed' || t.status === 'cancelled') {
                clearInterval(miningPollingRef.current!);
                miningPollingRef.current = null;
              }
            }
          } catch {
            // ignore
          }
        }, 5000);
      } catch (err: any) {
        console.error('Failed to start mining task:', err);
        alert(`任务启动失败 / Error: ${err.message || 'Unknown network error'}\n后端引擎连接中断或返回了错误的数据格式，模拟功能已屏蔽。`);
      }
    },
    [],
  );

  const startMining = useCallback(
    (config: TaskConfig) => {
      (async () => {
        const reachable = backendAvailable === true ? true : await probeBackendHealth();
        if (reachable) {
          startRealMining(config);
          return;
        }
        alert('错误：后端核心服务 (Port 8080) 无响应。为保障比赛规则评估和数据调度的真实有效性，系统已切断离线环境下的 Mock 模拟功能！\n\n请您使用 ./start_all.sh 脚本挂载后端接口后再继续。');
      })();
    },
    [backendAvailable, probeBackendHealth, startRealMining],
  );

  // Stop mining
  const stopMining = useCallback(async () => {
    if (!miningTask) return;
    if (miningPollingRef.current) {
      clearInterval(miningPollingRef.current);
      miningPollingRef.current = null;
    }
    if (backendAvailable) {
      try {
        await apiCancelMining(miningTask.taskId);
      } catch {
        // ignore
      }
    }
    setMiningTask((prev) => (prev ? { ...prev, status: 'failed' } : prev));
  }, [miningTask, backendAvailable]);

  // Reset mining task
  const resetMiningTask = useCallback(() => {
    if (miningPollingRef.current) {
      clearInterval(miningPollingRef.current);
      miningPollingRef.current = null;
    }
    setMiningTask(null);
    setMiningEquityCurve([]);
    setMiningDrawdownCurve([]);
    setMiningIcTimeSeries([]);
  }, []);

  // ==================================================================
  // BACKTEST
  // ==================================================================
  const [backtestTask, setBacktestTask] = useState<BacktestTask | null>(null);
  const [backtestLogs, setBacktestLogs] = useState<LogEntry[]>([]);

  const backtestPollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Start backtest
  const startBacktestTask = useCallback(
    async (params: BacktestStartParams) => {
      setBacktestLogs([]);
      const resp = await apiStartBacktest(params);
      if (!resp.success || !resp.data) throw new Error(resp.error || 'Failed');

      const taskData = resp.data.task as unknown as BacktestTask;
      setBacktestTask(taskData);

      // Polling fallback
      backtestPollingRef.current = setInterval(async () => {
        try {
          const r = await getBacktestStatus(resp.data!.taskId);
          if (r.data?.task) {
            const t = r.data.task as unknown as BacktestTask;

            // Always sync progress from polling (in case WS missed updates)
            setBacktestTask((prev) => {
              if (!prev) return t;
              return {
                ...prev,
                status: t.status,
                progress: t.progress || prev.progress,
                metrics: (t.metrics && Object.keys(t.metrics).length > 0) ? t.metrics : prev.metrics,
                updatedAt: t.updatedAt,
              };
            });

            if (t.status === 'completed' || t.status === 'failed' || t.status === 'cancelled') {
              // Final update: sync task + logs from backend (in case WS missed some)
              setBacktestTask(t);
              if (t.logs && t.logs.length > 0) {
                setBacktestLogs(t.logs.slice(-500));
              }
              clearInterval(backtestPollingRef.current!);
              backtestPollingRef.current = null;
            }
          }
        } catch {
          // ignore
        }
      }, 5000);
    },
    [],
  );

  // Stop backtest
  const stopBacktestTask = useCallback(async () => {
    if (!backtestTask) return;
    if (backtestPollingRef.current) {
      clearInterval(backtestPollingRef.current);
      backtestPollingRef.current = null;
    }
    try {
      await apiCancelBacktest(backtestTask.taskId);
    } catch {
      // ignore
    }
    setBacktestTask((prev) => (prev ? { ...prev, status: 'cancelled' } : prev));
  }, [backtestTask]);

  // ==================================================================
  // Context value
  // ==================================================================
  const value: TaskContextValue = {
    backendAvailable,
    // Mining
    miningTask,
    miningEquityCurve,
    miningDrawdownCurve,
    miningIcTimeSeries,
    startMining,
    stopMining,
    resetMiningTask,

    // ---- Backtest ----
    backtestTask,
    backtestLogs,
    startBacktestTask,
    stopBacktestTask,
  };

  return <TaskContext.Provider value={value}>{children}</TaskContext.Provider>;
};

// ========================== Hook ==========================

export function useTaskContext(): TaskContextValue {
  const ctx = useContext(TaskContext);
  if (!ctx) throw new Error('useTaskContext must be used inside <TaskProvider>');
  return ctx;
}
