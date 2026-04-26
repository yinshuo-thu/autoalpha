export interface DevTimelineEntry {
  timestamp: string;
  title: string;
  summary: string;
  tags: string[];
  bullets: string[];
}

// 后续每次”修改并跑通”后，在这里追加一条记录即可；页面会按 timestamp 自动排序。
export const devTimeline: DevTimelineEntry[] = [
  {
    timestamp: '2026-04-24T23:30:00+08:00',
    title: 'Model Lab 整体因子输出卡片精简 + 相关性图直接可见',
    summary: '简化 Cell 4 只展示唯一 pq 文件（最佳模型），IC/IR/Score 使用 evaluate_submission_like 口径；pq 输出 vs 入模因子相关性图直接显示在卡片内（无需打开 Modal）；后端在 pq 导出后从实际 pq 文件重算输入因子相关性；EnsembleModal 因子贡献权重加小字说明。',
    tags: ['Model Lab', 'Frontend', 'Backend', 'UX'],
    bullets: [
      '后端：run_model_lab 在 pq 导出后调用 _compute_submit_factor_input_correlations，将 pq vs 入模因子相关性写入 best_model.input_factor_correlations（从实际 pq 文件计算，非滚动预测）。',
      '前端 Cell 4：简化为只展示最佳模型；官方 IC/IR/Score/TVR/PassGates 突出显示；直接内嵌 pq vs 入模因子相关性迷你条形图，蓝色=正相关 橙色=负相关。',
      '前端 EnsembleModal 因子贡献权重：标题下增加一行小字，解释数值含义（LightGBM 特征增益 / 线性模型系数绝对值）。',
    ],
  },
  {
    timestamp: '2026-04-24T22:30:00+08:00',
    title: '修复因子相关性时序图不全 & 热图乱序问题',
    summary: '彻底修复两个长期 bug：时序图因 validRunIds 二次过滤导致部分通过因子丢失；热图行列顺序依赖后端缓存顺序而非生成时序。后端同时修复 heatmap_stale 被 trend_stale 短路跳过的逻辑 bug。',
    tags: ['Frontend', 'Backend', 'Bugfix', 'Correlation'],
    bullets: [
      'frontend: buildFactorCorrelationTrend 移除 validRunIds 过滤——trend_rows 已由后端保证只含通过因子，前端不应二次过滤。',
      'frontend: CorrelationHeatmap 在渲染前按 factorOrdinalByRunId 重排行列，确保横纵轴始终是 created_at 时序顺序，不依赖后端缓存内部顺序。',
      'backend: load_factor_correlation_cache 修复逻辑 bug——当 heatmap_stale=True 时触发全量重建（含 trend_rows），而不是被 trend_stale 短路只刷新 trend_rows。',
    ],
  },
  {
    timestamp: '2026-04-24T21:30:00+08:00',
    title: 'Model Lab 指标对齐因子口径 + 全模型相关性展示',
    summary: '统一 Model Lab 的 IC/IR/Score/TVR 展示口径（IC 已×100）；所有模型（含非最优 LGB 等）现在均计算与有效因子库的相关性，并可在整体因子输出卡片中点击查看。',
    tags: ['Model Lab', 'Backend', 'Frontend'],
    bullets: [
      '后端：_evaluate_predictions 增加 daily_ic_ir / daily_rank_ic_ir；模型汇总新增 avg_ir、avg_daily_ic_bps、avg_daily_rank_ic_bps（×100 口径）。',
      '后端：新增 _compute_pred_series_all_factor_correlations，对所有 PassGates 因子计算与滚动测试预测序列的相关性——所有模型均可获取 all_factor_correlations，不再局限于最优模型。',
      '后端：_export_submit_ready_model_factor 返回值增加 IC/IR/Score/tvr/PassGates 官方指标；run_model_lab 将其写入最优模型的 submit_* 字段。',
      '前端：整体因子输出 Cell 4 改为展示所有模型（非最优也可点击查看相关性）；Modal 优先展示官方指标，滚动均值指标用×100 口径。',
    ],
  },
  {
    timestamp: '2026-04-24T20:00:00+08:00',
    title: 'Fix IC 分布 x 轴精度问题',
    summary: 'IC 分布直方图 tickLabel 由 1 位小数改为 2 位，避免 bin 中点四舍五入后误导最大值显示。',
    tags: ['Frontend', 'AutoAlpha', 'Bugfix'],
    bullets: [
      'buildHistogram 对 IC 的 digits 参数从 1 改为 2，tickLabel 和 tooltip label 均更精确。',
      '修复前：实际 max IC=1.18 时最后一个 bin 中点显示为 "1.1"，让人误读范围。',
      '修复后：显示 "1.07"，tooltip 也正确展示 "0.95 ~ 1.18" 的实际 bin 边界。',
    ],
  },
  {
    timestamp: '2026-04-24T18:20:00+08:00',
    title: 'DEV 时间线回填 RAG 机制细节',
    summary: '不再单独维护 RAG block，而是把当前真实运行中的 RAG 机制按时间线补回原有节点，包括 query-aware 召回、Embedding 回退、auto compact 和公式 prompt 的组装顺序。',
    tags: ['DEV Page', 'RAG', 'Frontend'],
    bullets: [
      '移除 DEV 页面中的独立 RAG ROADMAP 区块，避免历史记录和规划说明割裂。',
      '把 RAG 的实际落地内容补回 2026-04-22 各时间点：v2 包拆分、inspiration 检索、passing factor/query-aware recall、generation experience 和记忆回灌。',
      '补清楚当前正样本召回不是静态 top-K，而是先由 Stage-1 hypothesis 产出 query_text，再做 semantic retrieval；如果 embedding 不可用，就自动回退到 lexical similarity + score anchor。',
      '补清楚 prompt 不是一次性硬拼，而是按 passing_rag、failure summary、hypothesis、parent contrast、inspiration、negative families、generation experience 等 section 组装，再走 auto compact 控制总长度。',
      'CODEX_CHANGELOG 也同步去掉独立 RAG 总结段，改成按时间线记录真实实现。',
    ],
  },
  {
    timestamp: '2026-04-09T23:30:00+08:00',
    title: '前端雏形与首轮代码审查',
    summary: '建立 frontend 工程骨架，并开始对 AutoAlpha v2 做代码级与架构级检查。',
    tags: ['Frontend', 'Review', 'Bootstrap'],
    bullets: [
      '形成 Vite + React + TypeScript 页面与组件结构。',
      '整理 review_0409_24pm.md，记录 evaluator、研究循环、leaderboard、IC 单位等问题。',
    ],
  },
  {
    timestamp: '2026-04-10T01:10:00+08:00',
    title: 'LLM / API 连通性与最早期提交流程测试',
    summary: '把模型调用、HTTP 请求和提交流程样例先跑起来，验证基础链路可用。',
    tags: ['LLM', 'API', 'Submit'],
    bullets: [
      '新增多份 test_api_connectivity / test_httpx / test_requests_long / test_llm_factor_gen 脚本。',
      '开始生成最早期 submit 备份包与 patch 修补脚本。',
    ],
  },
  {
    timestamp: '2026-04-10T10:41:12+08:00',
    title: '项目初版发布',
    summary: '提交 Initial publish，保留核心研究、评估、因子与前端代码骨架。',
    tags: ['Git', 'Publish'],
    bullets: [
      '提交 c6f1c28。',
      '排除本地数据集与运行环境，清理出可发布源码版本。',
    ],
  },
  {
    timestamp: '2026-04-10T12:28:00+08:00',
    title: 'README / 架构图 / 端到端样例补齐',
    summary: '补了项目说明、架构图和最早一批 LLM 端到端提交样例。',
    tags: ['Docs', 'Architecture', 'E2E'],
    bullets: [
      '更新 README 与架构说明，补充比赛规则、评分公式和运行方式。',
      '产出 llm_e2e / t1 等端到端提交目录用于链路验证。',
    ],
  },
  {
    timestamp: '2026-04-10T16:45:00+08:00',
    title: '手动因子搜索工作流',
    summary: '建立 manual 因子批量搜索和报告输出能力，开始系统化手工研究。',
    tags: ['Manual Factors', 'Research', 'Report'],
    bullets: [
      '新增 manual/manual_factor_runner.py。',
      '测试 66 个手工候选因子，覆盖价格、VWAP、波动、区间和活跃度等 family。',
      '生成 MANUAL_FACTOR_REPORT 与对应 CSV 结果。',
    ],
  },
  {
    timestamp: '2026-04-11T09:55:00+08:00',
    title: 'submission-like 重算与手动因子提交产物',
    summary: '把手动因子重新按 submission-like 口径计算，并统一提交目录结构。',
    tags: ['Evaluator', 'Submission', 'Manual Factors'],
    bullets: [
      '新增 manual/recalc_submit_metrics.py。',
      '扩展 core/evaluator.py，补更贴近官方口径的结果输出。',
      '生成 manual/submit 下 26 个 submission-like 产物与 metadata。',
    ],
  },
  {
    timestamp: '2026-04-21T23:20:56+08:00',
    title: 'AutoAlpha v1 指标重算与因子卡片体系',
    summary: '引入 autoalpha 自动研究包，并逐步形成记录页与因子卡片展示能力。',
    tags: ['AutoAlpha v1', 'Metrics', 'Records'],
    bullets: [
      '新增 autoalpha 研究模块：LLM client、knowledge base、pipeline、loop、factor research 等。',
      '接入官方口径指标重算流程，并在记录页展示因子卡片。',
    ],
  },
  {
    timestamp: '2026-04-22T00:18:43+08:00',
    title: '切分为 AutoAlpha v1 / v2',
    summary: '将原自动研究包整理成更清晰的 v1 / v2 结构，并同步接入服务端；这一步也把后续 RAG 链路所需的知识库、灵感库和 prompt 生成职责拆到了清晰模块里。',
    tags: ['AutoAlpha v2', 'Refactor', 'Server'],
    bullets: [
      '提交 f16a81c，创建 clean AutoAlpha v2 package。',
      'server.py 接入新的 v2 API 路由与服务逻辑。',
      'README 调整为中英文分层说明。',
      '在 v2 中正式拆出 knowledge_base、inspiration_db、llm_client、loop 等模块：knowledge_base 负责 passing / failed / generation memory，inspiration_db 负责外部灵感语料，llm_client 负责把这些上下文拼成可控预算的 prompt。',
    ],
  },
  {
    timestamp: '2026-04-22T01:13:19+08:00',
    title: 'v2 灵感源扩展与前端主页面成型',
    summary: '把 inspirations、来源统计和前端主页面结构一起补齐，形成 AutoAlpha / Record / Ideas 主导航，也把“外部灵感检索层 + 知识库检索层”这两类 RAG 语料正式分开。',
    tags: ['Ideas', 'Frontend', 'Analytics'],
    bullets: [
      '扩展 inspiration_db、inspiration_fetcher、knowledge_base、llm_client、pipeline。',
      '前端强化 AutoAlphaPage、AutoAlphaRecordsPage、InspirationBrowserPage 与 layout 导航。',
      'inspiration_db 开始承担独立的外部灵感检索层，不再只把 prompt 文本当静态素材，而是把 paper、笔记、prompt brainstorm 等来源转成可采样的 research context。',
      'knowledge_base 持续记录公式、thought process、父代、来源、过 gate 状态和失败细节，为 passing retrieval、negative family retrieval 和 generation lesson retrieval 提供统一语料。',
      '这时的思路已经形成分层：外层先从 inspiration 取“机制灵感”，内层再从 knowledge base 取“已验证结构、失败模式和对照样本”，共同服务最后一条公式 prompt。',
    ],
  },
  {
    timestamp: '2026-04-22T08:21:38+08:00',
    title: '长循环稳定性强化',
    summary: '围绕长期运行的 AutoAlpha loop 做状态保护、配置集中化和服务启动稳定性增强。',
    tags: ['Loop', 'Runtime', 'Stability'],
    bullets: [
      '新增 runtime_config.py，集中管理运行时配置。',
      '增强 start / stop / status 检测与服务端保护逻辑。',
      '前端补充运行态提示与控制细节。',
    ],
  },
  {
    timestamp: '2026-04-22T08:37:35+08:00',
    title: 'generation experience 汇总',
    summary: '把生成经验写入知识库并回灌到下一轮 prompt，让循环具备可积累的经验记忆；当前 RAG 也从“只看 passing factor”扩展成“正样本 + 失败经验 + 历史代际总结”的混合记忆。',
    tags: ['RAG', 'Prompt', 'Experience'],
    bullets: [
      'knowledge_base 增加 generation experience summary：每代会沉淀 tested、passing、best_score、failure_counts 和摘要文本，形成 generation 级别的实验复盘。',
      'llm_client / loop 接入最近 generation 经验上下文，默认把最近几代 summary 直接拼进 prompt，优先让模型感知最近的失败分布与修正方向。',
      '额外增加 relevance match：当前 archetype / dominant failure mode 会触发一次轻量关键词检索，把较早但相关的 generation lesson 也召回进来，避免经验只停留在最近 2 到 3 轮。',
      '记录页增加生成经验和代际统计展示。',
      '当前公式生成 prompt 的组成顺序也在这一阶段稳定下来：passing_rag -> failure_summary -> hypothesis_context -> parent contrast -> inspiration_text -> exhausted_families -> productive_pairs/crowded_tokens -> generation_experience -> mode_rules。',
      '为了不让上下文越积越大，prompt 会在发送前跑 auto compact：先按 section 大小做本地压缩，只在必要时对少数可压缩 section 做一次便宜 LLM 压缩，最后仍超预算就继续 deterministic trim，保证关键信息优先保留。',
    ],
  },
  {
    timestamp: '2026-04-22T23:48:08+08:00',
    title: '仓库修改日志归档',
    summary: '整理出 CODEX_CHANGELOG，开始把历史提交和当前工作区修改统一汇总。',
    tags: ['Changelog', 'Archive', 'Repo State'],
    bullets: [
      '按 git log、git diff、文件修改时间归纳历史改动。',
      '明确记录 AutoAlpha v2、LLM 挖掘、评估与前端/API 的持续增强方向。',
      '把 RAG 机制按时间线拆回对应节点，而不是额外维护一份脱离上下文的说明块。',
    ],
  },
  {
    timestamp: '2026-04-24T12:00:00+08:00',
    title: '新增 DEV 页面与模型 IC 对比精简',
    summary: '增加独立 DEV 时间线页面，并把 Model Lab 的模型对比图从 IC / TVR 改为只保留 IC 相关对比。',
    tags: ['DEV Page', 'Timeline', 'Frontend'],
    bullets: [
      '新增顶栏 DEV 导航与独立页面路由。',
      '首版时间线按 git log + CODEX_CHANGELOG 录入历史修改。',
      'AutoAlpha 页面中的”模型 IC / TVR 对比”改为只展示 Avg IC 与 Avg Rank IC。',
    ],
  },
  {
    timestamp: '2026-04-24T16:00:00+08:00',
    title: 'DEV 页面倒序展示 + 入口移至设置页 + v2 整包提交',
    summary: 'DEV 页面改为最新记录在上方；DEV 入口从顶栏移至任务设置页按钮区；对 v2 所有积累修改做首次完整 git push。',
    tags: ['DEV Page', 'Git', 'Navigation', 'Requirement'],
    bullets: [
      'DEV 时间线倒序显示，序号从最新往旧递减，方便快速查看最近动态。',
      'DEV 入口从顶栏导航移到任务设置页，与"测试 LLM"、"保存配置"等按钮并列，顶栏导航保持简洁。',
      '将 autoalpha_v2 全量重构、前端组件、后端 API、leaderboard、评估器等累积修改一次性 commit 并 push。',
      '新增 CLAUDE.md，写入 v2 项目要求：每次修改跑通后需同步更新 DEV 记录并 git push。',
    ],
  },
  {
    timestamp: '2026-04-24T03:35:00+08:00',
    title: '修复 IdeaCache 未命中导致高频 LLM 错误',
    summary: '定位到 pipeline 在 join_fill() 返回后未再次尝试 pop 缓存，导致每轮仍发起内联 LLM 请求，引发并发超时；新增两个防御措施彻底消除内联 LLM 调用。',
    tags: ['BugFix', 'IdeaCache', 'LLM', 'Pipeline'],
    bullets: [
      'pipeline.py：join_fill() 之后补充 cache-after-fill drain，fill 完成的 4 个 idea 现在会被直接 pop，不再触发内联 LLM。',
      'idea_cache.py：新增 join_fill() 方法，强制 pipeline 等待后台 fill 线程结束再继续。',
      'idea_cache.py：get_default_cache() 默认并发数从 2 降至 1，避免多路 LLM 请求同时打到 relay 触发 60s 超时。',
      '验证：重启 loop 后 Round 1 两个 idea 均显示 "from cache"，零内联 LLM 调用，零网络报错。',
    ],
  },
  {
    timestamp: '2026-04-24T09:45:00+08:00',
    title: '安装 scikit-learn/lightgbm 修复 Rolling Model Lab',
    summary: '修复 rolling_model_lab 因缺少 scikit-learn、scipy、lightgbm 而无法启动的问题；验证 30 轮连续挖掘可正常触发 Model Lab 更新流程。',
    tags: ['BugFix', 'ModelLab', 'Dependencies'],
    bullets: [
      '安装 scipy（force-reinstall 修复 sparse 子模块缺失）、scikit-learn 1.8.0、lightgbm 4.6.0 到 .venv。',
      '验证：新 loop 运行后 WARN 消失，Model Lab 将在每 10 个测试因子后自动触发。',
      '全流程确认：30 轮新 loop Round 1 测试 2/2 过关，Round 2 也全部来自 cache，零内联 LLM。',
      '当前累计 KB: 1930 tested / 58 passing / best=209.28。',
    ],
  },
  {
    timestamp: '2026-04-24T20:30:00+08:00',
    title: 'AutoAlpha 页 Rolling Model Lab 改为 2×2 布局并新增整体因子卡片',
    summary: '把 Rolling Model Lab 区块的四个内容面板重新排列为 2×2 网格，并为整体因子输出新增可点击的卡片和详情弹窗（含因子相关性、贡献权重、可填回测结果）。',
    tags: ['Frontend', 'AutoAlpha', 'ModelLab', 'Layout'],
    bullets: [
      '2×2 布局：左上=跨模型特征重要性共识，右上=特征重要性/预测摘要，左下=入模因子清单（限高 260px），右下=整体因子输出。',
      '整体因子输出说明更新为实际机制：Rolling Model Lab 只保存最佳模型的 pq 文件，与 ensemble_outputs 字段一致。',
      '新增 EnsembleModal 弹窗：可查看合成因子路径、模型表现指标、因子贡献权重 bar chart、入模因子两两相关性列表，以及可本地存储的回测结果表单。',
      '修复编辑引入的 U+201D 弯引号问题（184 处 className 属性引号被 editor 错误替换）；去除未使用的 FileStack import。',
    ],
  },
];
