# AutoAlpha v2 RAG 改进待办

> 基于对当前 RAG 机制的完整分析，列出 5 个优先改进方向，按优先级排序。

---

## 问题 1 ⭐ 正样本检索缺乏语义相关性（最高优先级）

### 现状
`compose_passing_factors_rag()` 直接取 `Score DESC` 的全局 top-12 注入 prompt，与当前生成方向完全无关。例如当 Stage-1 hypothesis 是"成交量耗竭反转"，但 top-12 里全是动量因子，这 12 条对本次生成没有指导价值，反而占用大量 token。

### 文件
- `knowledge_base.py` → `compose_passing_factors_rag()`
- `llm_client.py` → `generate_idea()` 调用处

### 改进方案
1. 在 `add_factor()` 写回时，对每条 passing factor 的 `thought_process` + `formula` 做 embedding，存入 KB（可存到 `knowledge.json` 的 `"embeddings"` 字段，或单独的 `embeddings.npy`）
2. Stage-1 生成 hypothesis 后，对 hypothesis 文本做同一 embedding
3. 用余弦相似度从 passing factors 中检索 top-K（K=6）最相关的，替代全局 top-12
4. 保留 1-2 条全局最高分因子作为"锚点"，防止完全偏离已验证结构

### 接口建议
```python
# knowledge_base.py
def compose_passing_factors_rag(self, query_text: str, k: int = 6) -> str:
    # 用 query_text 做语义检索，返回相关 passing factors
    ...
```

### 注意
- Embedding 模型推荐：`text-embedding-3-small`（便宜，延迟低）
- 写回时异步执行 embedding，不阻塞主评估流程
- 若 embedding 服务不可用，fallback 到当前 Score DESC 逻辑

---

## 问题 2 Inspiration 质量分没有反馈闭环

### 现状
`quality_score` 在 inspiration 入库时一次性写死，之后从不更新。某条 inspiration 被反复采样但对应生成的因子全部失败，它的分值仍然不变，下次还会被优先选中。

### 文件
- `inspiration_db.py` → `InspirationDB` 类
- `pipeline.py` → 评估完成后写回逻辑（`kb.add_factor()` 附近）

### 改进方案
1. `inspirations` 表新增两列：`usage_count INTEGER DEFAULT 0`、`pass_count INTEGER DEFAULT 0`
2. 每次 inspiration 被采样时，`usage_count += 1`
3. 因子评估结果写回 KB 时，同步更新对应 inspiration 的 `pass_count`（通过 `inspiration_ids` 字段关联）
4. 动态 quality_score 计算：`effective_score = base_quality_score * 0.5 + (pass_count / max(usage_count, 1)) * 0.5`
5. 新 inspiration（`usage_count == 0`）给予探索加成（避免冷启动全被旧 inspiration 压制）

### 接口建议
```python
# inspiration_db.py
def record_usage(self, inspiration_ids: list[int]) -> None: ...
def record_pass(self, inspiration_ids: list[int]) -> None: ...
def get_effective_score(self, row: dict) -> float: ...
```

---

## 问题 3 Exhausted Family 指纹粒度太粗

### 现状
`formula_structural_fingerprint()` 把所有字段统一替换为 `_f`，导致 `ts_mean(close, 5)` 和 `ts_mean(vwap, 10)` 被归为同一家族。实际上"价格类"和"量价混合类"的因子逻辑完全不同，不应被同一个"死路"标签屏蔽。

### 文件
- `knowledge_base.py` → `formula_structural_fingerprint()`

### 改进方案
将字段分为 3 个语义类别，指纹中保留类别标签而非全部替换为 `_f`：

```python
_PRICE_FIELDS = {"close_trade_px", "open_trade_px", "high_trade_px", "low_trade_px", "vwap"}
_VOLUME_FIELDS = {"volume", "dvolume", "trade_count"}
_MIXED_FIELDS  = {"amount", "bid_ask_spread", ...}

# 替换规则：
# close_trade_px → _price
# volume         → _vol
# trade_count    → _vol
# vwap           → _price  （或单独一类 _vwap，看需求）
```

这样 `ts_mean(close, _n)` → `ts_mean(_price, _n)`，`ts_mean(volume, _n)` → `ts_mean(_vol, _n)`，两者属于不同家族。

### 注意
- 修改后旧的 `family_records` 键会失效，需要在 `_load()` 时做一次性迁移或清空
- 可在 KB `version` 从 2 升到 3 时触发迁移

---

## 问题 4 Generation Experience 只取最近 3 轮，无语义检索

### 现状
`compose_recent_generation_experience_context()` 直接取最近 3 个 generation 的 summary 前 220 字拼到 prompt 末尾。当 generation 数积累到 10+，早期的关键教训被完全丢弃。若第 5 代曾解决了某类 TVR 问题，第 15 代遇到同类问题时完全不知道。

### 文件
- `knowledge_base.py` → `compose_recent_generation_experience_context()`
- `llm_client.py` → `generate_idea()` 注入点

### 改进方案
**方案 A（轻量，推荐先做）**：在拼 prompt 时，除了最近 2 轮，额外加一步：
- 扫描所有历史 summary，用关键词匹配当前 `archetype` 和 `failure_dominant_mode`
- 若有匹配，插入最多 1 条"历史相关教训"

```python
def find_relevant_experience(self, archetype: str, failure_mode: str) -> str | None:
    # 关键词匹配：archetype 名词 + failure_mode 类型
    # 返回最相关的 1 条 generation summary（前 300 字）
    ...
```

**方案 B（完整）**：同问题 1，对 generation summary 做 embedding，用当前 hypothesis 检索。

---

## 问题 5 Stage-1 Hypothesis 失败无反馈

### 现状
Stage-2（formula 生成）失败（syntax_error / compute_error）后，没有任何信号反馈给 Stage-1。下次同样的 archetype 还会生成类似的 hypothesis，导致重复踩坑。

### 文件
- `idea_cache.py` → `IdeaCache` 表结构
- `pipeline.py` → idea 消费和结果写回逻辑
- `llm_client.py` → `_generate_hypothesis()` 的 archetype 选择逻辑

### 改进方案
1. `idea_cache` 表新增 `outcome TEXT DEFAULT NULL`（取值：`syntax_error` / `compute_error` / `screened_out` / `passing`）
2. 因子评估后，通过 `idea_id` 回写 outcome 到 idea_cache 表
3. 在 `generate_idea()` 的 archetype 选择时，统计各 archetype 的历史 outcome 分布：
   ```python
   # 若某 archetype 最近 10 次中 syntax_error + compute_error > 5，降低其采样权重
   ```
4. 这个统计可以很轻量，不需要 embedding，只是 `GROUP BY archetype, outcome COUNT(*)` 查询

---

## 实现顺序建议

| 顺序 | 问题 | 估计工作量 | 预期收益 |
|------|------|-----------|---------|
| 1 | 问题 3：指纹粒度 | 1-2h | 立即减少误判死路 |
| 2 | 问题 2：inspiration 反馈 | 2-3h | 逐渐提升 inspiration 采样质量 |
| 3 | 问题 5：Stage-1 反馈 | 2-3h | 减少 syntax/compute 错误率 |
| 4 | 问题 4-A：经验关键词匹配 | 1-2h | 不增加依赖，快速改善历史召回 |
| 5 | 问题 1：语义检索 | 4-6h | 最高收益，但需引入 embedding API |

---

*文档由 Claude 生成，日期：2026-04-23*
