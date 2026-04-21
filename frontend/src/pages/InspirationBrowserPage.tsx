import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  BookOpen,
  Brain,
  ExternalLink,
  Globe,
  Layers,
  Loader2,
  MessageSquare,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  ToggleLeft,
  ToggleRight,
  Zap,
} from 'lucide-react';

// ─── Types ────────────────────────────────────────────────────────────────────

interface Inspiration {
  id: number;
  kind: string;
  title: string;
  source: string;
  content: string;
  summary: string;
  tags: string;
  source_type: string;
  arxiv_id: string;
  published_date: string;
  quality_score: number;
  status: 'active' | 'inactive';
  created_at: string;
}

interface PaginatedResult {
  items: Inspiration[];
  total: number;
  page: number;
  per_page: number;
  pages: number;
}

interface CacheStatus {
  pending: number;
  consumed: number;
  total: number;
  fill_running: boolean;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const json = await res.json();
  if (!res.ok || json.success === false) {
    throw new Error(json.error || json.detail || `API 请求失败 (${res.status})`);
  }
  if (json.data !== undefined) return json.data as T;
  return json as T;
}

const SOURCE_TYPE_LABELS: Record<string, { label: string; color: string; icon: React.FC<{ className?: string }> }> = {
  arxiv:  { label: 'ArXiv',  color: 'bg-blue-500/20 text-blue-300 border-blue-500/30',   icon: BookOpen },
  wechat: { label: 'WeChat', color: 'bg-green-500/20 text-green-300 border-green-500/30', icon: MessageSquare },
  llm:    { label: 'LLM',    color: 'bg-purple-500/20 text-purple-300 border-purple-500/30', icon: Brain },
  future: { label: 'Future', color: 'bg-orange-500/20 text-orange-300 border-orange-500/30', icon: BookOpen },
  manual: { label: '手动',   color: 'bg-amber-500/20 text-amber-300 border-amber-500/30',  icon: Layers },
  url:    { label: 'URL',    color: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',    icon: Globe },
};

function SourceBadge({ type }: { type: string }) {
  const cfg = SOURCE_TYPE_LABELS[type] || SOURCE_TYPE_LABELS.manual;
  const Icon = cfg.icon;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs border ${cfg.color}`}>
      <Icon className="h-3 w-3" />
      {cfg.label}
    </span>
  );
}

function formatDate(iso: string) {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso.slice(0, 10);
  return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric', year: 'numeric' });
}

// ─── Card ─────────────────────────────────────────────────────────────────────

const InspirationCard: React.FC<{
  item: Inspiration;
  onToggle: (id: number) => void;
  onDelete: (id: number) => void;
}> = ({ item, onToggle, onDelete }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={`glass rounded-xl border transition-all duration-200 ${
        item.status === 'inactive' ? 'opacity-40 border-border/30' : 'border-border/50 hover:border-primary/30'
      }`}
    >
      <div className="p-4">
        {/* Header row */}
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap mb-1">
              <SourceBadge type={item.source_type} />
              {item.published_date && (
                <span className="text-xs text-muted-foreground">{item.published_date}</span>
              )}
              {item.arxiv_id && (
                <span className="text-xs font-mono text-muted-foreground">{item.arxiv_id}</span>
              )}
            </div>
            <h3 className="text-sm font-semibold leading-snug truncate" title={item.title}>
              {item.title}
            </h3>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-1 shrink-0">
            {item.source && item.source.startsWith('http') && (
              <a
                href={item.source}
                target="_blank"
                rel="noreferrer"
                className="p-1 rounded hover:bg-secondary/50 text-muted-foreground hover:text-foreground transition-colors"
                title="打开链接"
              >
                <ExternalLink className="h-3.5 w-3.5" />
              </a>
            )}
            <button
              onClick={() => onToggle(item.id)}
              className="p-1 rounded hover:bg-secondary/50 text-muted-foreground hover:text-foreground transition-colors"
              title={item.status === 'active' ? '停用' : '启用'}
            >
              {item.status === 'active'
                ? <ToggleRight className="h-3.5 w-3.5 text-green-400" />
                : <ToggleLeft className="h-3.5 w-3.5" />}
            </button>
            <button
              onClick={() => onDelete(item.id)}
              className="p-1 rounded hover:bg-red-500/20 text-muted-foreground hover:text-red-400 transition-colors"
              title="删除"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* Summary */}
        <p className="mt-2 text-xs text-muted-foreground leading-relaxed line-clamp-2">
          {item.summary || item.content.slice(0, 200)}
        </p>

        {/* Expand */}
        {item.content && item.content.length > 200 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="mt-1 text-xs text-primary/70 hover:text-primary transition-colors"
          >
            {expanded ? '收起' : '展开全文'}
          </button>
        )}
        {expanded && (
          <pre className="mt-2 text-xs text-muted-foreground whitespace-pre-wrap bg-secondary/30 rounded p-2 max-h-60 overflow-y-auto">
            {item.content}
          </pre>
        )}

        {/* Footer */}
        <div className="mt-2 flex items-center justify-between">
          <span className="text-xs text-muted-foreground/60">{formatDate(item.created_at)}</span>
          {item.tags && (
            <div className="flex gap-1 flex-wrap">
              {item.tags.split(',').map((t) => t.trim()).filter(Boolean).map((tag) => (
                <span key={tag} className="text-xs px-1.5 py-0.5 rounded bg-secondary/40 text-muted-foreground">
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// ─── Add Form ─────────────────────────────────────────────────────────────────

const AddInspirationForm: React.FC<{ onAdded: () => void }> = ({ onAdded }) => {
  const [input, setInput] = useState('');
  const [title, setTitle] = useState('');
  const [sourceType, setSourceType] = useState('manual');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const isUrl = /^https?:\/\//i.test(input.trim());

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    setLoading(true);
    setError('');
    try {
      await apiFetch('/api/autoalpha/inspirations', {
        method: 'POST',
        body: JSON.stringify({
          input: input.trim(),
          raw_input: input.trim(),
          title: title.trim(),
          source_type: isUrl ? (input.includes('mp.weixin.qq.com') ? 'wechat' : sourceType) : sourceType,
        }),
      });
      setInput('');
      setTitle('');
      onAdded();
    } catch (exc: any) {
      setError(String(exc.message || exc));
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="glass rounded-xl border border-border/50 p-4 space-y-3">
      <div className="flex items-center gap-2 mb-1">
        <Plus className="h-4 w-4 text-primary" />
        <span className="text-sm font-medium">添加灵感</span>
      </div>

      <div>
        <input
          type="text"
          placeholder="粘贴 URL（微信文章、arxiv 等）或输入文字想法…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="w-full bg-secondary/30 border border-border/50 rounded-lg px-3 py-2 text-sm
                     placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary/50"
        />
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          placeholder="标题（可选）"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          className="flex-1 bg-secondary/30 border border-border/50 rounded-lg px-3 py-2 text-sm
                     placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary/50"
        />
        <select
          value={sourceType}
          onChange={(e) => setSourceType(e.target.value)}
          className="bg-secondary/30 border border-border/50 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary/50"
        >
          <option value="manual">手动</option>
          <option value="wechat">微信</option>
          <option value="url">URL</option>
          <option value="llm">LLM</option>
          <option value="arxiv">ArXiv</option>
          <option value="future">Future</option>
        </select>
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium
                     hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : '添加'}
        </button>
      </div>

      {error && <p className="text-xs text-red-400">{error}</p>}
    </form>
  );
};

// ─── Cache Status Bar ─────────────────────────────────────────────────────────

const CacheStatusBar: React.FC = () => {
  const [status, setStatus] = useState<CacheStatus | null>(null);

  const refresh = useCallback(async () => {
    try {
      const data = await apiFetch<CacheStatus>('/api/autoalpha/idea-cache/status');
      setStatus(data);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    refresh();
    const iv = setInterval(refresh, 10_000);
    return () => clearInterval(iv);
  }, [refresh]);

  if (!status) return null;

  return (
    <div className="flex items-center gap-3 px-3 py-2 glass rounded-lg border border-border/40 text-xs">
      <Zap className={`h-3.5 w-3.5 ${status.fill_running ? 'text-yellow-400 animate-pulse' : 'text-muted-foreground'}`} />
      <span className="text-muted-foreground">想法缓存:</span>
      <span className="font-medium">{status.pending} 待用</span>
      <span className="text-muted-foreground">/</span>
      <span className="text-muted-foreground">{status.consumed} 已用</span>
      {status.fill_running && (
        <span className="text-yellow-400 animate-pulse">生成中…</span>
      )}
    </div>
  );
};

// ─── Main Page ────────────────────────────────────────────────────────────────

export const InspirationBrowserPage: React.FC = () => {
  const [data, setData] = useState<PaginatedResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [sourceType, setSourceType] = useState('all');
  const [search, setSearch] = useState('');
  const [includeInactive, setIncludeInactive] = useState(false);
  const [fetchingMore, setFetchingMore] = useState(false);
  const [fetchMessage, setFetchMessage] = useState('');
  const [fetchError, setFetchError] = useState('');
  const searchRef = useRef<ReturnType<typeof setTimeout>>();

  // Stable async loader — takes all params explicitly (no closure over state).
  const load = useCallback(async (p: number, st: string, q: string, inactive: boolean) => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        page: String(p),
        per_page: '20',
        source_type: st,
        search: q,
        include_inactive: String(inactive),
      });
      const result = await apiFetch<PaginatedResult>(`/api/autoalpha/inspirations/browse?${params}`);
      setData(result);
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  }, []);

  // Single debounced effect — avoids stale-closure bugs.
  // search gets 400ms debounce; filter/page changes are instant.
  const prevSearchRef = useRef(search);
  useEffect(() => {
    const isSearchChange = search !== prevSearchRef.current;
    prevSearchRef.current = search;
    const delay = isSearchChange ? 400 : 0;
    clearTimeout(searchRef.current);
    searchRef.current = setTimeout(
      () => load(isSearchChange ? 1 : page, sourceType, search, includeInactive),
      delay,
    );
    return () => clearTimeout(searchRef.current);
  }, [page, sourceType, search, includeInactive, load]);

  const handleToggle = async (id: number) => {
    try {
      await apiFetch(`/api/autoalpha/inspirations/${id}/toggle`, { method: 'PUT' });
      load(page, sourceType, search, includeInactive);
    } catch { /* ignore */ }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm('确认删除该灵感？')) return;
    try {
      await apiFetch(`/api/autoalpha/inspirations/${id}`, { method: 'DELETE' });
      load(page, sourceType, search, includeInactive);
    } catch { /* ignore */ }
  };

  const handleFetchMore = async () => {
    setFetchingMore(true);
    setFetchMessage('');
    setFetchError('');
    try {
      const result = await apiFetch<{ added?: Record<string, number> }>('/api/autoalpha/inspirations/fetch?llm_ideas=6&arxiv_per_query=5', {
        method: 'POST',
      });
      const added = result.added || {};
      setFetchMessage(`抓取完成：ArXiv ${added.arxiv ?? 0}，URL ${added.url ?? 0}，LLM ${added.llm ?? 0}，Future ${added.future ?? 0}，重复 ${added.skipped ?? 0}`);
      setPage(1);
      await load(1, sourceType, search, includeInactive);
    } catch (exc: any) {
      setFetchError(String(exc.message || exc || '抓取失败'));
    } finally {
      setFetchingMore(false);
    }
  };

  const sourceTypes = ['all', 'arxiv', 'future', 'wechat', 'llm', 'manual', 'url'];

  return (
    <div className="max-w-4xl mx-auto space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">灵感库</h2>
          <p className="text-sm text-muted-foreground mt-0.5">
            管理 ArXiv 论文、公众号文章、LLM 构想等多源因子灵感
          </p>
        </div>
        <CacheStatusBar />
      </div>

      {/* Add form */}
      <AddInspirationForm onAdded={() => { setPage(1); load(1, sourceType, search, includeInactive); }} />

      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Search */}
        <div className="relative flex-1 min-w-48">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <input
            type="text"
            placeholder="搜索标题、摘要…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-3 py-2 bg-secondary/30 border border-border/50 rounded-lg text-sm
                       placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary/50"
          />
        </div>

        {/* Source type filter */}
        <div className="flex gap-1">
          {sourceTypes.map((st) => (
            <button
              key={st}
              onClick={() => { setSourceType(st); setPage(1); }}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                sourceType === st
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-secondary/30 text-muted-foreground hover:text-foreground hover:bg-secondary/60'
              }`}
            >
              {st === 'all' ? '全部' : (SOURCE_TYPE_LABELS[st]?.label ?? st)}
            </button>
          ))}
        </div>

        {/* Include inactive toggle */}
        <button
          onClick={() => setIncludeInactive(!includeInactive)}
          className={`text-xs px-3 py-1.5 rounded-lg transition-all ${
            includeInactive ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                            : 'bg-secondary/30 text-muted-foreground hover:bg-secondary/60'
          }`}
        >
          含停用
        </button>

        {/* Fetch more */}
        <button
          onClick={handleFetchMore}
          disabled={fetchingMore}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
                     bg-primary/10 text-primary border border-primary/20 hover:bg-primary/20 transition-all
                     disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {fetchingMore
            ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
            : <RefreshCw className="h-3.5 w-3.5" />}
          抓取新灵感
        </button>
      </div>

      {(fetchMessage || fetchError) && (
        <div className={`rounded-lg border px-3 py-2 text-xs ${
          fetchError
            ? 'border-red-500/30 bg-red-500/10 text-red-300'
            : 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
        }`}>
          {fetchError || fetchMessage}
        </div>
      )}

      {/* Stats row */}
      {data && (
        <div className="text-xs text-muted-foreground">
          共 <span className="font-medium text-foreground">{data.total}</span> 条灵感
          {data.total > 0 && `，第 ${data.page}/${data.pages} 页`}
        </div>
      )}

      {/* List */}
      {loading ? (
        <div className="flex items-center justify-center py-16 text-muted-foreground gap-2">
          <Loader2 className="h-5 w-5 animate-spin" />
          <span className="text-sm">加载中…</span>
        </div>
      ) : data?.items.length === 0 ? (
        <div className="text-center py-16 text-muted-foreground">
          <BookOpen className="h-10 w-10 mx-auto mb-3 opacity-30" />
          <p className="text-sm">暂无灵感。点击「抓取新灵感」或手动添加。</p>
        </div>
      ) : (
        <div className="space-y-3">
          {data?.items.map((item) => (
            <InspirationCard
              key={item.id}
              item={item}
              onToggle={handleToggle}
              onDelete={handleDelete}
            />
          ))}
        </div>
      )}

      {/* Pagination */}
      {data && data.pages > 1 && (
        <div className="flex items-center justify-center gap-2 pt-2">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="px-3 py-1.5 rounded-lg text-xs bg-secondary/30 text-muted-foreground
                       hover:bg-secondary/60 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
          >
            上一页
          </button>
          {Array.from({ length: Math.min(data.pages, 7) }, (_, i) => {
            const pg = i + Math.max(1, page - 3);
            if (pg > data.pages) return null;
            return (
              <button
                key={pg}
                onClick={() => setPage(pg)}
                className={`w-8 h-8 rounded-lg text-xs transition-all ${
                  pg === page
                    ? 'bg-primary text-primary-foreground font-medium'
                    : 'bg-secondary/30 text-muted-foreground hover:bg-secondary/60'
                }`}
              >
                {pg}
              </button>
            );
          })}
          <button
            onClick={() => setPage((p) => Math.min(data.pages, p + 1))}
            disabled={page >= data.pages}
            className="px-3 py-1.5 rounded-lg text-xs bg-secondary/30 text-muted-foreground
                       hover:bg-secondary/60 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
          >
            下一页
          </button>
        </div>
      )}
    </div>
  );
};
