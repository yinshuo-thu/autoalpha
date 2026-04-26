export interface DevRagPriority {
  priority: number;
  title: string;
  summary: string;
  currentState: string;
  files: string[];
  actions: string[];
  notes?: string[];
  effort: string;
  impact: string;
}

export interface DevRagExecutionStep {
  order: number;
  title: string;
  effort: string;
  impact: string;
}

export const devRagMeta = {
  title: 'AutoAlpha v2 RAG 改进路线图',
  subtitle: '把后端仓库里的 RAG 改进待办同步到 DEV 页面，方便前后端一起对齐当前问题、落地顺序和收益预期。',
  source: 'autoalpha_v2/RAG_TODO.md',
  updatedAt: '2026-04-23',
};

export const devRagPriorities: DevRagPriority[] = [
  {
    priority: 1,
    title: '正样本检索缺乏语义相关性',
    summary: '当前 passing factor RAG 直接取全局高分 top-12，和本轮 hypothesis 的方向可能完全不相干，既浪费 token，也容易把生成带偏。',
    currentState: 'compose_passing_factors_rag() 仍以 Score DESC 为主，没有根据当前 hypothesis / archetype 做语义召回。',
    files: ['knowledge_base.py', 'llm_client.py'],
    actions: [
      '在 passing factor 写回知识库时，为 thought_process + formula 生成 embedding。',
      'Stage-1 生成 hypothesis 后，用同一 embedding 将当前 query_text 编码。',
      '按余弦相似度召回 top-K 相关 passing factors，替代纯全局排序。',
      '保留 1 到 2 条最高分因子作为锚点，避免检索结果完全偏离已验证结构。',
    ],
    notes: [
      'Embedding 服务不可用时需要回退到当前 Score DESC 逻辑。',
      '推荐异步写 embedding，避免阻塞主评估流程。',
    ],
    effort: '4-6h',
    impact: '最高收益，直接提升 prompt 中正样本的命中率。',
  },
  {
    priority: 2,
    title: 'Inspiration 质量分没有反馈闭环',
    summary: '灵感源的 quality_score 目前是一锤子买卖，被反复采样但持续失败的 inspiration 不会降权，导致抽样越来越不聪明。',
    currentState: 'inspiration 入库后没有 usage_count / pass_count 反馈，effective score 不会随真实表现更新。',
    files: ['inspiration_db.py', 'pipeline.py'],
    actions: [
      '为 inspirations 增加 usage_count 和 pass_count 字段。',
      '每次 inspiration 被采样时记录 usage_count。',
      '通过因子评估结果回写对应 inspiration 的 pass_count。',
      '将 effective_score 改为 base_quality_score 与 pass rate 的混合分数，并给新 inspiration 冷启动探索加成。',
    ],
    effort: '2-3h',
    impact: '逐轮改善灵感采样质量，减少无效 inspiration 的重复进入。',
  },
  {
    priority: 3,
    title: 'Exhausted Family 指纹粒度太粗',
    summary: '结构指纹把字段统一替换成 `_f`，会把价格类和成交量类等本质不同的因子误判成同一家族，过早标记为死路。',
    currentState: 'formula_structural_fingerprint() 对字段做过度抽象，家族判定缺少语义类别。',
    files: ['knowledge_base.py'],
    actions: [
      '把字段按 price / volume / mixed 等语义类别映射成不同占位符。',
      '让 ts_mean(_price, _n) 与 ts_mean(_vol, _n) 进入不同 family。',
      '结合 KB version 升级，在加载时迁移或清空旧 family_records。',
    ],
    notes: ['修改后旧 family_records 键值会失效，需要一次性迁移策略。'],
    effort: '1-2h',
    impact: '最快见效，能够立刻减少误判死路带来的检索与探索损失。',
  },
  {
    priority: 4,
    title: 'Generation Experience 只取最近 3 轮',
    summary: '当前 generation 经验上下文只拼最近 3 条 summary，历史中真正有价值的教训无法在后续同类问题里被召回。',
    currentState: 'compose_recent_generation_experience_context() 没有基于 archetype / failure mode 的相关性检索。',
    files: ['knowledge_base.py', 'llm_client.py'],
    actions: [
      '先做轻量版：除最近 2 轮外，再按 archetype 与 failure_dominant_mode 做关键词匹配。',
      '把最相关的 1 条历史经验插回 prompt。',
      '后续可进一步升级为 embedding 检索，与 passing factor RAG 使用相同语义检索链路。',
    ],
    effort: '1-2h',
    impact: '低成本提升“历史教训可复用度”，尤其适合 TVR / coverage 等重复失败模式。',
  },
  {
    priority: 5,
    title: 'Stage-1 Hypothesis 失败无反馈',
    summary: '如果 Stage-2 在 syntax_error / compute_error 失败，Stage-1 并不会收到负反馈，因此同类 archetype 会重复踩坑。',
    currentState: 'idea_cache 没有完整 outcome 字段，也没有按 archetype 统计失败分布后反向调权。',
    files: ['idea_cache.py', 'pipeline.py', 'llm_client.py'],
    actions: [
      '在 idea_cache 增加 outcome 字段，记录 syntax_error / compute_error / screened_out / passing。',
      '评估完成后通过 idea_id 回写 outcome。',
      '在 generate_idea() 选择 archetype 时参考最近 outcome 分布，对高失败 archetype 降权。',
    ],
    effort: '2-3h',
    impact: '减少结构性重复失败，降低 syntax / compute 错误率。',
  },
];

export const devRagExecutionPlan: DevRagExecutionStep[] = [
  {
    order: 1,
    title: 'Exhausted Family 指纹粒度',
    effort: '1-2h',
    impact: '立即减少误判死路',
  },
  {
    order: 2,
    title: 'Inspiration 反馈闭环',
    effort: '2-3h',
    impact: '逐步提升 inspiration 采样质量',
  },
  {
    order: 3,
    title: 'Stage-1 失败反馈',
    effort: '2-3h',
    impact: '减少 syntax / compute 错误',
  },
  {
    order: 4,
    title: '经验关键词检索',
    effort: '1-2h',
    impact: '快速提升历史经验召回',
  },
  {
    order: 5,
    title: 'Passing Factor 语义检索',
    effort: '4-6h',
    impact: '最高收益，但依赖 embedding 能力',
  },
];
