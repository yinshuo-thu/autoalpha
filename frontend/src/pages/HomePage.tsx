import React, { useMemo, useState } from 'react';
import { ShieldCheck, Trophy, FileCheck2, Clock3 } from 'lucide-react';
import { ProgressSidebar } from '@/components/ProgressSidebar';
import { LiveCharts } from '@/components/LiveCharts';
import { ChatInput } from '@/components/ChatInput';
import { FactorStatsRow } from '@/components/FactorStatsRow';
import { FactorList } from '@/components/FactorList';
import { useTaskContext } from '@/context/TaskContext';
import { Layout } from '@/components/layout/Layout';
import type { PageId } from '@/components/layout/Layout';
import { executeFactor, healthCheck } from '@/services/api';
import { getDefaultMiningDirection } from '@/utils/miningDirections';
import { formatNumber } from '@/utils';
import { setActiveLibraryName } from '@/utils/autoalphaStorage';

interface HomePageProps {
  onNavigate?: (page: PageId) => void;
}

const ScoreFormula = () => (
  <div className="glass rounded-2xl border border-border/60 p-5">
    <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">竞赛评分核心</div>
    <div className="mt-3 text-2xl md:text-3xl font-semibold text-foreground">
      Score = (IC - 0.0005 x TVR) x sqrt(IR) x 100
    </div>
    <p className="mt-3 text-sm leading-6 text-muted-foreground">
      只有先通过 Gate，Score 才有意义。页面上的排序、摘要和提交提示都已经统一按这个口径展示。
    </p>
  </div>
);

export const HomePage: React.FC<HomePageProps> = ({ onNavigate }) => {
  const {
    backendAvailable,
    miningTask: task,
    miningEquityCurve: equityCurve,
    miningDrawdownCurve: drawdownCurve,
    startMining,
    stopMining,
  } = useTaskContext();
  const [manualFactor, setManualFactor] = useState<any | null>(null);
  const [manualLoading, setManualLoading] = useState(false);

  const manualGateEntries = useMemo(() => {
    const gatePayload = manualFactor?.gates_detail || manualFactor?.gatesDetail;
    if (!gatePayload) return [];
    return Object.entries(gatePayload as Record<string, boolean>);
  }, [manualFactor]);

  const handleGenerate = async (config: { userInput: string; useCustomMiningDirection?: boolean }) => {
    if (backendAvailable === false) {
      try {
        await healthCheck();
      } catch {
        alert('后端未连接，无法执行指定因子生成。请先启动 8080 后端服务。');
        return;
      }
    }
    const input = config.useCustomMiningDirection ? getDefaultMiningDirection() : config.userInput.trim();
    if (!input) {
      alert('请输入自然语言方向或 DSL 公式。');
      return;
    }
    setManualLoading(true);
    try {
      const resp = await executeFactor({ input });
      if (!resp.success || !resp.data?.factor) {
        throw new Error(resp.error || '生成失败');
      }
      setManualFactor(resp.data.factor);
    } catch (error: any) {
      alert(`指定因子生成失败: ${error.message || '未知错误'}`);
    } finally {
      setManualLoading(false);
    }
  };

  const renderBlankState = () => (
    <div className="flex flex-col min-h-[60vh] pb-8 animate-fade-in-up">
      <div className="mb-6 flex justify-between items-end border-b border-border/50 pb-4">
        <div>
          <h2 className="text-3xl font-semibold text-foreground">AutoAlpha</h2>
          <p className="text-muted-foreground mt-2 text-sm">
            指定生成、自动挖掘、真实评估、提交导出已经串成一条链路，重点围绕 Score 和可提交状态工作。
          </p>
          {backendAvailable === false && (
            <p className="text-sm text-warning mt-3">后端未连接，当前只能浏览页面，不能执行真实因子计算。</p>
          )}
          {backendAvailable === true && (
            <p className="text-sm text-success mt-3">后端已连接，可直接生成因子并导出到 submit 目录。</p>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1.2fr_0.8fr] gap-4 mb-4">
        <ScoreFormula />
        <div className="glass rounded-2xl border border-border/60 p-5">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">提交格式</div>
          <div className="mt-3 text-lg font-semibold text-foreground">submit/因子名_时间戳_y|n</div>
          <p className="mt-3 text-sm leading-6 text-muted-foreground">
            `y` 表示质量 Gate 和提交格式都通过，`n` 表示仍需修复。输出为单个 `.pq` 文件，路径和元数据会直接显示在结果卡片中。
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 mb-4">
        <div className="glass rounded-2xl border border-border/60 p-5 lg:col-span-2">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Trophy className="h-4 w-4 text-amber-500" />
            指定生成
          </div>
          <div className="mt-3 text-xl font-semibold">自然语言方向 or DSL 公式</div>
          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            自然语言会先生成候选公式，再跑真实 quick test；直接输入 DSL 会立刻算分、校验频率和导出提交文件。
          </p>
        </div>
        <div className="glass rounded-2xl border border-border/60 p-5">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <ShieldCheck className="h-4 w-4 text-emerald-500" />
            质量 Gate
          </div>
          <div className="mt-3 text-sm leading-7 text-foreground">IC / IR / TVR / 集中度 / Coverage</div>
        </div>
        <div className="glass rounded-2xl border border-border/60 p-5">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Clock3 className="h-4 w-4 text-sky-500" />
            时间栅格
          </div>
          <div className="mt-3 text-sm leading-7 text-foreground">UTC 15 分钟频率，严格按引擎交易时间导出</div>
        </div>
      </div>

      {(manualLoading || manualFactor) && (
        <div className="glass rounded-2xl border border-border/60 p-5 mb-4">
          <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4 mb-4">
            <div>
              <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">指定生成结果</div>
              <div className="mt-2 text-2xl font-semibold text-foreground">
                {manualLoading ? '正在生成并验证因子...' : manualFactor?.factor_name || '最新结果'}
              </div>
              {!manualLoading && (
                <div className="mt-2 text-sm text-muted-foreground">
                  {manualFactor?.source_mode && (
                    <span className="mr-2 rounded-md bg-primary/15 px-2 py-0.5 text-xs text-primary">
                      来源: {String(manualFactor.source_mode)}
                    </span>
                  )}
                  {manualFactor?.recommendation || manualFactor?.reason || '已完成真实评估（quick_test +  leaderboard）'}
                </div>
              )}
            </div>
            {!manualLoading && manualFactor?.submission_path && (
              <div className="flex gap-2">
                <button
                  className="rounded-xl bg-primary px-4 py-2 text-sm text-primary-foreground"
                  onClick={() => onNavigate?.('records')}
                >
                  打开 Record
                </button>
              </div>
            )}
          </div>

          {manualLoading ? (
            <p className="text-sm text-muted-foreground">后台正在加载缓存并计算真实指标，这一步通常需要几十秒到数分钟。</p>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div className="rounded-2xl bg-amber-500/10 border border-amber-500/20 p-4">
                  <div className="text-xs text-muted-foreground">Score</div>
                  <div className="mt-2 font-mono text-2xl font-semibold text-amber-400">
                    {formatNumber(manualFactor?.Score ?? 0, 4)}
                  </div>
                </div>
                <div className="rounded-2xl bg-secondary/20 p-4">
                  <div className="text-xs text-muted-foreground">IC</div>
                  <div className="mt-2 font-mono text-lg font-semibold">{formatNumber(manualFactor?.IC ?? 0, 4)}</div>
                </div>
                <div className="rounded-2xl bg-secondary/20 p-4">
                  <div className="text-xs text-muted-foreground">IR</div>
                  <div className="mt-2 font-mono text-lg font-semibold">{formatNumber(manualFactor?.IR ?? 0, 4)}</div>
                </div>
                <div className="rounded-2xl bg-secondary/20 p-4">
                  <div className="text-xs text-muted-foreground">Turnover</div>
                  <div className="mt-2 font-mono text-lg font-semibold">{formatNumber(manualFactor?.Turnover ?? 0, 4)}</div>
                </div>
                <div
                  className={`rounded-2xl border p-4 ${
                    manualFactor?.submission_ready_flag
                      ? 'bg-emerald-500/10 border-emerald-500/20'
                      : 'bg-rose-500/10 border-rose-500/20'
                  }`}
                >
                  <div className="text-xs text-muted-foreground">可直接提交</div>
                  <div className="mt-2 text-lg font-semibold">
                    {manualFactor?.submission_ready_flag ? 'YES' : 'NO'}
                  </div>
                </div>
              </div>

              <div className="rounded-2xl bg-secondary/20 p-4 border border-border/50">
                <div className="text-xs text-muted-foreground mb-2">公式</div>
                <code className="text-sm break-all">{manualFactor?.formula}</code>
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-[1fr_1fr] gap-4">
                <div className="rounded-2xl bg-background/40 p-4 border border-border/50">
                  <div className="flex items-center gap-2 text-sm font-medium">
                    <ShieldCheck className="h-4 w-4 text-emerald-500" />
                    Gate 明细
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {manualGateEntries.length > 0 ? (
                      manualGateEntries.map(([key, value]) => (
                        <span
                          key={key}
                          className={`rounded-full border px-3 py-1 text-xs ${
                            value
                              ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
                              : 'border-rose-500/30 bg-rose-500/10 text-rose-400'
                          }`}
                        >
                          {key}
                        </span>
                      ))
                    ) : (
                      <span className="text-sm text-muted-foreground">暂无 Gate 数据</span>
                    )}
                  </div>
                </div>

                <div className="rounded-2xl bg-background/40 p-4 border border-border/50">
                  <div className="flex items-center gap-2 text-sm font-medium">
                    <FileCheck2 className="h-4 w-4 text-sky-500" />
                    提交校验
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <div className="text-muted-foreground">Coverage</div>
                      <div className="mt-1 font-semibold">
                        {manualFactor?.sanity_report?.cover_all ? 'PASS' : 'FAIL'}
                      </div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Exact Grid</div>
                      <div className="mt-1 font-semibold">
                        {manualFactor?.sanity_report?.exact_15m_grid ? 'PASS' : 'FAIL'}
                      </div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Missing Days</div>
                      <div className="mt-1 font-semibold">{manualFactor?.sanity_report?.missing_days_count ?? '--'}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Rows</div>
                      <div className="mt-1 font-semibold">{manualFactor?.sanity_report?.row_count ?? '--'}</div>
                    </div>
                  </div>
                </div>
              </div>

              {manualFactor?.submission_path && (
                <div className="rounded-2xl bg-emerald-500/5 p-4 border border-emerald-500/20">
                  <div className="text-xs text-muted-foreground mb-1">提交文件</div>
                  <div className="text-sm break-all text-foreground">{manualFactor.submission_path}</div>
                  {manualFactor?.metadata_path && (
                    <div className="mt-2 text-xs break-all text-muted-foreground">{manualFactor.metadata_path}</div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <details className="glass group rounded-2xl border border-border/60 bg-secondary/15 p-4 open:bg-secondary/25 transition-colors" open>
          <summary className="flex cursor-pointer items-center justify-between font-semibold text-foreground select-none">
            可用高频字段
            <span className="text-muted-foreground transform group-open:rotate-180 transition-transform">▼</span>
          </summary>
          <div className="mt-4 text-sm text-muted-foreground leading-7">
            `close_trade_px`、`open_trade_px`、`high_trade_px`、`low_trade_px`、`volume`、`dvolume`、`trade_count`、`vwap`
            都可以直接作为特征输入。
            <div className="mt-3 text-rose-400">禁止引用 `resp`、`trading_restriction` 等未来信息字段。</div>
          </div>
        </details>

        <details className="glass group rounded-2xl border border-border/60 bg-secondary/15 p-4 open:bg-secondary/25 transition-colors" open>
          <summary className="flex cursor-pointer items-center justify-between font-semibold text-foreground select-none">
            引擎合规算子
            <span className="text-muted-foreground transform group-open:rotate-180 transition-transform">▼</span>
          </summary>
          <div className="mt-4 text-sm text-muted-foreground leading-7">
            `ts_mean`、`ts_std`、`ts_sum`、`ts_max`、`ts_min`、`delta`、`lag`、`ts_rank`、`ts_zscore`、
            `ts_decay_linear`、`cs_rank`、`cs_zscore`、`cs_demean`、`safe_div`、`signed_power`
          </div>
        </details>
      </div>
    </div>
  );

  return (
    <Layout
      currentPage="home"
      onNavigate={onNavigate || (() => {})}
      showNavigation={!!onNavigate}
    >
      {!task ? (
        renderBlankState()
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 animate-fade-in">
          {task.engineMeta && task.engineMeta.llmEnabled === false && (
            <div className="lg:col-span-4 rounded-2xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
              当前运行时配置中<strong>未检测到 LLM API Key</strong>：自动挖掘将主要使用进化算法与离线库，不会在界面中「假装」大模型输出。
              请到设置页填写 Key 并点击「测试 LLM」通过后重试。
            </div>
          )}
          <div className="lg:col-span-1">
            <ProgressSidebar progress={task.progress} />
          </div>
          <div className="lg:col-span-3">
            <LiveCharts
              equityCurve={equityCurve}
              drawdownCurve={drawdownCurve}
              metrics={task.metrics || null}
              isRunning={task.status === 'running'}
              logs={task.logs}
              llmMiningRecent={task.llmMiningRecent}
              logPaths={task.logPaths}
            />
          </div>

          <div className="lg:col-span-4">
            <FactorStatsRow
              metrics={task.metrics || null}
              onBacktest={() => {
                if (task.config?.librarySuffix) {
                  const libName = `all_factors_library_${task.config.librarySuffix}.json`;
                  setActiveLibraryName(libName);
                } else {
                  setActiveLibraryName('all_factors_library.json');
                }
                onNavigate?.('backtest');
              }}
            />
          </div>
          <div className="lg:col-span-4">
            <FactorList metrics={task.metrics || null} />
          </div>
        </div>
      )}

      <ChatInput
        onSubmit={startMining}
        onGenerate={handleGenerate}
        onStop={stopMining}
        isRunning={task?.status === 'running'}
        isGenerating={manualLoading}
      />
    </Layout>
  );
};
