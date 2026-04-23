import React from 'react';
import { Clock3, FileCode2, GitBranch, History } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { devTimeline } from '@/data/devTimeline';

function formatTimelineDate(timestamp: string) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return timestamp;
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}

export const DevPage: React.FC = () => {
  const totalEntries = devTimeline.length;
  const firstEntry = devTimeline[0];
  const latestEntry = devTimeline[devTimeline.length - 1];
  const displayTimeline = [...devTimeline].reverse();

  return (
    <div className="space-y-6">
      <section className="rounded-[32px] border border-border/60 bg-[linear-gradient(135deg,rgba(15,23,42,0.96)_0%,rgba(15,118,110,0.90)_55%,rgba(8,145,178,0.88)_100%)] p-6 text-white shadow-[0_28px_80px_rgba(15,23,42,0.18)]">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs tracking-[0.18em] text-white/85">
              <History className="h-3.5 w-3.5" />
              DEV TIMELINE
            </div>
            <h2 className="mt-4 text-3xl font-semibold tracking-tight">开发演进记录</h2>
            <p className="mt-3 max-w-2xl text-sm leading-7 text-white/82">
              最新修改在最上方。每次功能修改并验证跑通后，在 `frontend/src/data/devTimeline.ts`
              追加一条记录，同步 git push。
            </p>
          </div>
          <div className="grid min-w-[260px] gap-3 sm:grid-cols-3">
            <div className="rounded-2xl border border-white/15 bg-white/10 p-4">
              <div className="text-xs uppercase tracking-[0.16em] text-white/70">记录数</div>
              <div className="mt-2 text-2xl font-semibold">{totalEntries}</div>
            </div>
            <div className="rounded-2xl border border-white/15 bg-white/10 p-4">
              <div className="text-xs uppercase tracking-[0.16em] text-white/70">起点</div>
              <div className="mt-2 text-sm font-medium">{firstEntry ? formatTimelineDate(firstEntry.timestamp) : '--'}</div>
            </div>
            <div className="rounded-2xl border border-white/15 bg-white/10 p-4">
              <div className="text-xs uppercase tracking-[0.16em] text-white/70">最新</div>
              <div className="mt-2 text-sm font-medium">{latestEntry ? formatTimelineDate(latestEntry.timestamp) : '--'}</div>
            </div>
          </div>
        </div>
      </section>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.65fr)]">
        <div className="space-y-4">
          {displayTimeline.map((entry, index) => (
            <div key={`${entry.timestamp}-${entry.title}`} className="relative pl-8">
              <div className="absolute left-3 top-0 h-full w-px bg-gradient-to-b from-sky-200 via-teal-200 to-transparent" />
              <div className="absolute left-0 top-6 flex h-6 w-6 items-center justify-center rounded-full border border-sky-200 bg-white text-[11px] font-semibold text-sky-700 shadow-sm">
                {totalEntries - index}
              </div>
              <Card className="border border-border/60 bg-white/90 shadow-[0_18px_50px_rgba(15,23,42,0.06)]">
                <CardHeader className="pb-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
                        {formatTimelineDate(entry.timestamp)}
                      </div>
                      <CardTitle className="mt-2 text-xl text-foreground">{entry.title}</CardTitle>
                      <p className="mt-3 text-sm leading-7 text-muted-foreground">{entry.summary}</p>
                    </div>
                    <div className="flex flex-wrap justify-end gap-2">
                      {entry.tags.map((tag) => (
                        <span
                          key={`${entry.title}-${tag}`}
                          className="rounded-full border border-sky-200 bg-sky-50 px-3 py-1 text-xs font-medium text-sky-700"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {entry.bullets.map((bullet) => (
                      <div key={bullet} className="rounded-2xl bg-slate-50 px-4 py-3 text-sm leading-7 text-slate-700">
                        {bullet}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          ))}
        </div>

        <div className="space-y-4">
          <Card className="border border-border/60 bg-white/90">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <FileCode2 className="h-4 w-4 text-sky-600" />
                维护规则
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm leading-7 text-muted-foreground">
              <div className="rounded-2xl bg-slate-50 px-4 py-3">
                每次功能修改并验证通过后，在 `frontend/src/data/devTimeline.ts` 追加一条新记录。
              </div>
              <div className="rounded-2xl bg-slate-50 px-4 py-3">
                一条记录至少写时间、标题、简述、标签和 2 到 4 条关键改动。
              </div>
              <div className="rounded-2xl bg-slate-50 px-4 py-3">
                优先记录“结构变化、链路打通、评估口径变化、前端可见变化”，避免只写零碎样式调整。
              </div>
            </CardContent>
          </Card>

          <Card className="border border-border/60 bg-white/90">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <GitBranch className="h-4 w-4 text-emerald-600" />
                首版来源
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm leading-7 text-muted-foreground">
              <div className="rounded-2xl bg-emerald-50 px-4 py-3 text-emerald-800">
                `git log --reverse` 用来梳理关键提交时间点。
              </div>
              <div className="rounded-2xl bg-emerald-50 px-4 py-3 text-emerald-800">
                `CODEX_CHANGELOG.md` 用来补未提交阶段的主题性改动摘要。
              </div>
            </CardContent>
          </Card>

          <Card className="border border-border/60 bg-white/90">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Clock3 className="h-4 w-4 text-amber-600" />
                当前状态
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm leading-7 text-muted-foreground">
              <div className="rounded-2xl bg-amber-50 px-4 py-3 text-amber-900">
                这版 DEV 页面已经可以作为单独日志页使用，后续继续按时间追加即可。
              </div>
              <div className="rounded-2xl bg-amber-50 px-4 py-3 text-amber-900">
                本次也同步把 Model Lab 的模型对比图聚焦到 IC 指标，不再混入 TVR。
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
