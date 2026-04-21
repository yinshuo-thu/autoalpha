import React, { useEffect, useMemo, useState } from 'react';
import { FileStack, GitBranch } from 'lucide-react';
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

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4" onClick={onClose}>
      <div className="max-h-[88vh] w-full max-w-4xl overflow-y-auto rounded-[28px] border border-border/50 bg-white p-6 shadow-2xl" onClick={(event) => event.stopPropagation()}>
        <div className="mb-5 flex items-center justify-between gap-3">
          <div>
            <div className="text-[11px] uppercase tracking-[0.22em] text-muted-foreground">LOG</div>
            <div className="mt-2 text-xl font-semibold text-foreground">{runId}</div>
          </div>
          <button onClick={onClose} className="rounded-full border border-border/60 px-3 py-1 text-sm text-muted-foreground transition-colors hover:text-foreground">
            关闭
          </button>
        </div>
        {error ? <div className="rounded-2xl bg-red-50 p-4 text-sm text-red-600">{error}</div> : null}
        {!report && !error ? <div className="text-sm text-muted-foreground">LOG 加载中...</div> : null}
        {report ? (
          <div className="space-y-4">
            <div className="rounded-3xl border border-border/50 bg-slate-50 p-4">
              <div className="text-xs text-muted-foreground">公式</div>
              <pre className="mt-2 whitespace-pre-wrap break-all text-xs text-slate-700">{report.formula}</pre>
            </div>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
              {['IC', 'IR', 'Turnover', 'Score', 'PassGates'].map((key) => (
                <div key={key} className="rounded-2xl border border-border/50 bg-white p-3">
                  <div className="text-[11px] text-muted-foreground">{key}</div>
                  <div className="mt-2 font-semibold text-foreground">{String(report.metrics?.[key] ?? '--')}</div>
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
      <Panel title="AutoAlpha 记录库" subtitle="Generation 演进、产出文件、LOG 和知识库因子表">
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
                              <div className="truncate font-mono text-xs text-foreground">{factor.run_id}</div>
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

      <Panel title="文件与 LOG 留存">
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
            <div className="mb-3 text-sm font-medium text-foreground">LOG 索引</div>
            <div className="max-h-[460px] space-y-3 overflow-y-auto pr-2">
              {researchReports.length === 0 ? (
                <div className="text-sm text-muted-foreground">当前还没有 LOG 文件。</div>
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
          <table className="min-w-[1620px] table-fixed text-sm">
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
                <th className="w-20 px-3 py-3 text-center">LOG</th>
              </tr>
            </thead>
            <tbody>
              {factors.length === 0 ? (
                <tr>
                  <td colSpan={13} className="px-3 py-12 text-center text-sm text-muted-foreground">
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
                        <div className="break-all font-mono text-xs text-foreground">{factor.run_id}</div>
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
                      <td className="px-3 py-3 text-center align-top">
                        {factor.live_submitted && factor.live_test_result ? (
                          <HoverCard>
                            <HoverCardTrigger asChild>
                              <button onClick={() => openLiveResultModal(factor)} className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-medium text-emerald-700 transition-colors hover:bg-emerald-200">
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
                                        <div className="mt-1 font-mono text-foreground">{String(metrics[key] ?? '--')}</div>
                                      </div>
                                    );
                                  })}
                                </div>
                                <pre className="max-h-44 overflow-auto whitespace-pre-wrap break-all rounded-xl bg-slate-50 p-3 font-mono text-[11px] leading-5 text-slate-700">{factor.live_test_result.raw}</pre>
                              </div>
                            </HoverCardContent>
                          </HoverCard>
                        ) : (
                          <button onClick={() => openLiveResultModal(factor)} className="rounded-full border border-border/60 px-3 py-1 text-xs text-foreground transition-colors hover:bg-white">
                            填入
                          </button>
                        )}
                      </td>
                      <td className="px-3 py-3 text-center">
                        {factor.research_path ? (
                          <button onClick={() => setSelectedRunId(factor.run_id)} className="rounded-full border border-border/60 px-3 py-1 text-xs text-foreground transition-colors hover:bg-white">
                            查看
                          </button>
                        ) : (
                          <span className="text-xs text-muted-foreground">-</span>
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
