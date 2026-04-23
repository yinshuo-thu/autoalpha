# Auto Alpha Research Factory v2

[English](README.en.md) | [中文](README.zh-CN.md) | [语言入口](README.md)

这是一个用于从干净状态重新启动 AutoAlpha 的代码版 v2。运行态状态和生成产物
被有意排除在代码说明之外：不依赖已存在的 `knowledge.json`，不把 parquet
输出、研究报告、提交文件、模型缓存或本地数据库视为源码的一部分。

![Auto Alpha Research Factory v2](v2.png)

AutoAlpha 是当前工作区里的因子研究流水线。它负责生成日内 alpha 想法，校验
DSL 公式，计算 15 分钟频率 alpha 文件，用类官方指标做评估，并维护知识库和
通过门槛后的可提交产物。

## v2 技术思想

AutoAlpha v2 是一个闭环的日内因子工厂。核心思想是让 agent 持续提出紧凑的
DSL 公式，然后强制每一个公式通过同一条确定性链路：语法与合规校验、带 warmup
的因子计算、类官方指标评估、提交网格导出、研究诊断、知识库持久化，以及下一代
父因子选择。

流水线刻意区分 **idea 质量** 和 **提交就绪状态**：

- idea agent 只输出 `formula`、`thought_process`、`postprocess` 和
  `lookback_days`，不直接写文件，也不自行决定因子是否优秀。
- `formula_validator.py` 和算子注册表定义合法搜索空间，阻断不支持的字段、算子
  和明显未来函数。
- `pipeline.compute_alpha` 在评估窗口前加载 warmup 历史，让滚动算子初始化充分，
  然后再裁剪回真实评估日期。
- 近期窗口筛选会跳过显著弱信号，避免把全历史计算资源浪费在低质量候选上；有希望
  的候选才会在完整区间重新计算。
- `core.evaluator.evaluate_submission_like_wide` 是 IC、IR、换手、集中度、覆盖、
  gate 和 score 的单一可信来源。
- 只有通过所有 gate 的因子才会复制到 `autoalpha_v2/submit`，并在研究 UI 中显示
  为可提交卡片链接。
- 通过因子会进入 `knowledge.json`、结构指纹、算子组合记忆和前端记录页，成为后续
  agent 的父因子和示例。

实践中，v2 像一个自主研究台：agent 产生假设，平台级 evaluator 做严格审稿，知识层
记住有效模式和已经耗尽的公式族。

## 与 v0 的区别

仓库根目录仍保留较早的 v0 风格研究栈：手写因子脚本、EA/LLM 循环、通用 DSL 解析、
leaderboard、回测辅助、提交工具和大量探索产物。AutoAlpha v2 更窄、更严格：它保留
有用的项目级基础设施，但封装成一个可审计、可追踪来源、可产出提交候选的研究工厂。

| 领域 | v0 根目录工作流 | AutoAlpha v2 |
|------|-----------------|--------------|
| 研究模式 | 脚本化实验：`research_loop.py`、`evaluate_alpha.py`、手动配置和 leaderboard 更新。 | 产品化循环：`autoalpha_v2/run.py` / `loop.py` 负责生成、筛选、导出、报告、通知和持久化。 |
| 指标对齐 | 有本地指标，但旧导出可能和平台假设偏离。 | 强制使用 15 分钟类官方 evaluator、交易限制后指标、修正 TVR、集中度 gate 和完整网格 parquet 校验。 |
| 产物策略 | 探索输出散落在 `outputs/`、`research/`、`submit/` 和手工报告中。 | 通过因子获得规范 `.pq`、metadata、类官方结果 JSON、报告和 factor card。 |
| 知识记忆 | leaderboard 和日志非正式地指导后续迭代。 | `knowledge.json` 存储所有测试因子、失败原因、父代血缘、指纹、卡片路径、Lab Test 结果和 generation 汇总。 |
| agent 反馈 | 可复用顶部公式，但失败族不够显式。 | LLM prompt 会接收强样例、近期弱样例、高产算子组合和饱和结构族。 |
| 灵感来源 | prompt 多为临时脚本输入。 | Paper、LLM brainstorm 和本地 `fut_feat/*.md` 期货因子笔记被统一同步到灵感库，并归因到生成因子。 |
| 前端 | 通用 dashboard/backend 集成。 | 专用 AutoAlpha cockpit：额度/状态、prompt lab、rolling model lab、generation 记录、提交卡片链接、灵感库和来源转化图。 |
| 提交安全 | 提交工具可以独立调用。 | 提交就绪是 gate 控制状态，只有 `PassGates=true` 因子会进入 `autoalpha_v2/submit`。 |

## 从 v1 保留的内容

v2 保留了 v1 中和平台规则对齐的因子导出与指标计算修复：

- alpha parquet 导出会在构建完整提交网格前规范化 `date`、`datetime` 和
  `security_id`，避免生成文件看似有效但实际大面积为空。
- 指标在应用 `trading_restriction` 后计算，匹配平台描述：受限证券在 IC、换手、
  book weight 和集中度计算前被移除。
- Turnover (`tvr`) 是日均逐 bar 原始 alpha 变化之和，并按当前绝对 alpha 规模归一化，
  展示为百分数点 (`x100`)。
- 持仓指标使用参考文本中的权重思路：
  `10000 * alpha_i / sum(abs(alpha))`。负值在 `bs`、`minn` 和 `min` 中保留负号。
- Gate 与 score 逻辑：
  `cover_all = 1`、`IC > 0.6`、`IR > 2.5`、`tvr < 400`、
  `maxx < 50`、`abs(minn) < 50`、`max < 20`、`abs(min) < 20`，
  然后 `score = (IC - 0.0005 * tvr) * sqrt(IR) * 100`。

## 目录结构

```text
autoalpha_v2/
├── llm_client.py              # LLM idea 生成
├── pipeline.py                # generate -> validate -> compute -> evaluate -> export
├── run.py                     # 新因子生成 CLI 入口
├── recompute_gate_factors.py  # 用当前指标逻辑重算旧通过因子
└── .gitignore                 # 防止运行态产物进入 Git
```

AutoAlpha 使用的项目级文件：

```text
core/evaluator.py      # 类官方指标实现
core/submission.py     # 完整网格 parquet 导出
prepare_data.py        # DataHub 数据加载器，读取 pv、response 和 restriction
start_all.sh           # 后端/前端启动器
frontend/              # React UI，包括 AutoAlpha records 页面
```

## 快速开始

从仓库根目录运行：

```bash
cd /Volumes/T7/Scientech

# 启动后端和前端。
./start_all.sh

# 复用已经健康运行的服务。
./start_all.sh --reuse

# 在完整评估区间生成新因子。
python autoalpha_v2/run.py --n 3

# 使用较短评估窗口快速迭代。
python autoalpha_v2/run.py --n 3 --days 120
```

服务地址：

- 后端：`http://127.0.0.1:8080`
- 前端：`http://127.0.0.1:3000`

在 macOS 上，`start_all.sh` 使用 `launchctl` LaunchAgents，让服务在启动命令结束后
继续运行。日志写入 `~/Library/Logs/Scientech`。

## 因子生成流程

1. `llm_client.py` 请求 LLM 输出 `formula`、`thought_process`、`postprocess` 和
   `lookback_days`。
2. `formula_validator.py` 拒绝不支持的字段/算子和明显未来函数。
3. `pipeline.compute_alpha` 在 `DataHub.pv_15m` 上计算公式，并在评估窗口前加载
   warmup 历史。
4. 后处理将原始信号变成适合提交的横截面，常用 `rank_clip` 到 `[-0.5, 0.5]`，或
   clipped z-score 变体。
5. `pipeline.evaluate_alpha` 使用 response 数据和交易限制调用
   `core.evaluator.evaluate_submission_like_wide`。
6. `pipeline.export_parquet` 写出包含 `date`、`datetime`、`security_id` 和 `alpha`
   的标准 `.pq` 文件。
7. `factor_research.analyze_factor` 生成研究报告。如果且仅当因子通过所有提交 gate，
   还会写出 `factor_card.json` 和 `factor_card.md`。
8. 通过因子会复制到 `autoalpha_v2/submit`，附带 metadata 和类官方结果 JSON；
   其 `run_id` 会成为 records 表中的可点击卡片链接。

## Factor Cards

Factor card 只为提交就绪因子生成，也就是 `PassGates=true`。被拒绝、重复、无效、
计算错误和 screened-out 的因子仍保留在研究日志和表格里，但不会生成卡片。

卡片目录：

```text
autoalpha_v2/research/<run_id>/
├── report.json
├── report.md
├── analysis.png
├── factor_card.json
└── factor_card.md
```

前端从 `knowledge.json` 读取 `factor_card_path`。在知识库表格中，只有可提交因子的
`run_id` 会显示为链接；点击后打开卡片/报告 modal。

卡片覆盖八个紧凑视角：

| 部分 | 内容 | 用途 |
|------|------|------|
| 因子定义 | 公式、输入字段、更新频率、预测目标、universe、postprocess。 | 解释因子捕捉什么，并辅助去重。 |
| 历史分布 | 直方图、P1/P5/P50/P95/P99、均值、标准差、偏度、峰度、缺失率、极端值比例。 | 识别偏态、尾部依赖和是否需要 clip/rank/zscore。 |
| 时间演化 | 日均值、日标准差、覆盖率、滚动漂移。 | 发现漂移、不稳定 regime 和覆盖问题。 |
| 预测能力 | IC 均值、ICIR、Rank IC、滚动 IC、horizon/lag IC。 | 判断因子是否有效以及在哪个 horizon 有效。 |
| 分层表现 | 分位收益、top-minus-bottom spread、累计 spread 曲线。 | 检查单调性和是否只靠尾部。 |
| 有效 regime | 高/低波动、趋势/震荡或可用 response regime 下的 IC。 | 判断因子适用环境和是否需要 gating。 |
| 稳定性 | 月度/年度 IC、train/val/test、裁剪尾部 IC。 | 测试是否依赖特定时期或异常值。 |
| 相关性与冗余 | 公式族、近似因子代理、alpha pool overlap、target-correlation proxy。 | 避免 alpha 池重复。 |

## DSL 公式语言

v1 DSL 被限制在当前/过去 15 分钟 bar 数据上，但算子集合已经足够支持更丰富搜索。

字段：

```text
open_trade_px, high_trade_px, low_trade_px, close_trade_px,
trade_count, volume, dvolume, vwap
```

时间序列算子：

```text
lag(x,d), delay(x,d), delta(x,d), ts_pct_change(x,d),
ts_mean(x,d), ts_ema(x,d), ts_std(x,d), ts_sum(x,d),
ts_max(x,d), ts_min(x,d), ts_median(x,d), ts_quantile(x,d,q),
ts_zscore(x,d), ts_rank(x,d), ts_minmax_norm(x,d),
ts_decay_linear(x,d), decay_linear(x,d),
ts_corr(x,y,d), ts_cov(x,y,d),
ts_skew(x,d), ts_kurt(x,d), ts_argmax(x,d), ts_argmin(x,d)
```

横截面算子：

```text
cs_rank(x), rank(x), cs_zscore(x), zscore(x),
cs_demean(x), demean(x), cs_scale(x), scale(x),
cs_winsorize(x,p), winsorize(x,p), cs_quantile(x,q),
cs_neutralize(x,y)
```

数学、条件与混合算子：

```text
safe_div(a,b), div(a,b), signed_power(x,p), pow(x,p),
abs(x), sign(x), neg(x), log(x), signed_log(x), sqrt(x),
clip(x,a,b), clamp(x,a,b), min_of(x,y), max_of(x,y),
sigmoid(x), tanh(x),
ifelse(cond,a,b), gt(x,y), ge(x,y), lt(x,y), le(x,y), eq(x,y),
and_op(a,b), or_op(a,b), not_op(a),
mean_of(x1,x2,...), weighted_sum(w1,x1,w2,x2,...),
combine_rank(x1,x2,...)
```

安全规则：

- lookback 参数如 `d` 必须是正整数常量。
- 禁止 `lead`、`future_*`、`resp` 和 `trading_restriction`。
- 条件算子允许使用，但可能提高换手；可用 `ts_mean`、`ts_ema` 或
  `ts_decay_linear` 平滑结果。
- 序列分母优先使用 `safe_div`，简单标量常数可用中缀 `/`。

## 类官方指标

evaluator 运行在平台的 15 分钟 bar 节奏上。如果 alpha 来自更高频数据，评估前只应保留
每个 15 分钟平台 bar 的最终值。

主要指标：

- `IC`：日均横截面 Pearson correlation，乘以 100。
- `IR`：日 IC 年化稳定性，`mean(daily_ic) / std(daily_ic) * sqrt(252)`。
- `tvr`：日均换手。逐 bar 公式为
  `sum(abs(alpha_t - alpha_t-1)) / sum(abs(alpha_t))`，日内求和后展示为 `x100`。
- `bl` / `bs`：每个 bar 的平均多头/空头 book weight。
- `nl` / `ns` / `nt`：每个 bar 的多头、空头和非零 alpha 股票数。
- `maxx` / `minn`：全区间单证券最大正/负 position weight，单位 bps。
- `max` / `min`：日均单证券最大/最小 position weight，单位 bps。
- `nd`：评估交易日数。
- `cover_all`：所有评估日都有覆盖时为 1。

Quality gates 和 score 同时写入 `knowledge.json` 和 submit metadata。前端直接读取这些值，
所以任何指标逻辑变更都应走下方重算路径。

## 重算提交候选

当指标或导出逻辑变化时，应重建旧候选，而不是直接信任旧 `knowledge.json`：

```bash
# 只重算曾经 PassGates=true 的因子。
python autoalpha_v2/recompute_gate_factors.py

# 更快：保留旧研究报告，只刷新 pq 和指标。
python autoalpha_v2/recompute_gate_factors.py --skip-research

# 重算 knowledge.json 中所有因子。
python autoalpha_v2/recompute_gate_factors.py --all --skip-research
```

脚本会：

- 备份 `autoalpha_v2/knowledge.json`；
- 归档旧 `autoalpha_v2/submit` 内容；
- 从源数据重算每个公式；
- 重新生成 `autoalpha_v2/output/<run_id>.pq`；
- 刷新 `knowledge.json` 中的指标和 gate 字段；
- 只把仍然通过 gate 的因子复制到 `autoalpha_v2/submit`；
- 在 `autoalpha_v2/recompute_reports` 下写出批处理摘要。

可提交因子是 `autoalpha_v2/submit` 目录下的 `.pq` 文件。每个文件都有对应 metadata JSON
和类官方结果 JSON，记录 UI 所使用的精确指标。

## 前端记录页

AutoAlpha records 页面展示：

- generation 血缘；
- output 文件和 factor-card 产物；
- 带 `Status/Gate` 与 `Lab Test` 列的因子表，factor ID 可直接打开卡片；
- 不同失败原因、未提交通过因子和已填写 Lab Test 的因子使用不同颜色。

Lab Test 结果可以粘贴到行 modal 中。它们会写回知识库，并和本地类官方指标分开展示。

## 前端

v2 records 页面增加了 Paper / LLM / Future 灵感来源转化图，包括来源数量、通过因子数、
通过率、每个 prompt 产生的有效因子数，以及各来源在全部有效因子中的占比。

## 常用命令

```bash
# 检查 recompute 脚本语法。
python -m py_compile autoalpha_v2/recompute_gate_factors.py

# 构建前端。
npm --prefix frontend run build

# 查看当前提交候选。
find autoalpha_v2/submit -maxdepth 1 -name '*.pq' -print

# 重算后启动服务。
./start_all.sh
```

## 运行说明

- 生成的 parquet 文件可能很大，视为运行态产物。Git 中保留代码、metadata、README 和
  recompute 摘要；除非发布明确需要二进制产物，否则不要提交大 `.pq` 文件。
- `AUTOALPHA_CLOUD_TVR_MULTIPLIER` 可用于给本地 score 应用保守换手乘数；在换手对齐
  类官方计算后，v1 默认值为 `1.0`。
- 如果本地 score 很好但 Lab Test 里 `score = 0`，优先比较
  `*_official_like_result.json`、parquet 文件大小和 `cover_all`。最常见失败模式是导出
  网格不完整或指标数据陈旧。
