export interface DevTimelineEntry {
  timestamp: string;
  title: string;
  summary: string;
  tags: string[];
  bullets: string[];
}

// 后续每次“修改并跑通”后，按时间顺序在这里追加一条记录即可。
export const devTimeline: DevTimelineEntry[] = [
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
    summary: '将原自动研究包整理成更清晰的 v1 / v2 结构，并同步接入服务端。',
    tags: ['AutoAlpha v2', 'Refactor', 'Server'],
    bullets: [
      '提交 f16a81c，创建 clean AutoAlpha v2 package。',
      'server.py 接入新的 v2 API 路由与服务逻辑。',
      'README 调整为中英文分层说明。',
    ],
  },
  {
    timestamp: '2026-04-22T01:13:19+08:00',
    title: 'v2 灵感源扩展与前端主页面成型',
    summary: '把 inspirations、来源统计和前端主页面结构一起补齐，形成 AutoAlpha / Record / Ideas 主导航。',
    tags: ['Ideas', 'Frontend', 'Analytics'],
    bullets: [
      '扩展 inspiration_db、inspiration_fetcher、knowledge_base、llm_client、pipeline。',
      '前端强化 AutoAlphaPage、AutoAlphaRecordsPage、InspirationBrowserPage 与 layout 导航。',
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
    summary: '把生成经验写入知识库并回灌到下一轮 prompt，让循环具备可积累的经验记忆。',
    tags: ['RAG', 'Prompt', 'Experience'],
    bullets: [
      'knowledge_base 增加 generation experience summary。',
      'llm_client / loop 接入最近 generation 经验上下文。',
      '记录页增加生成经验和代际统计展示。',
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
];
