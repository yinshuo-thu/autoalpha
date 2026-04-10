# Manual Factor Report

## Scope

本轮没有沿用 LLM 因子自动生成思路，而是单独建立了 `manual/` 路径，直接从 15 分钟价量数据中手工构造一批可解释因子，再用项目内的官方评估逻辑筛选有效因子。

本次搜索使用的输入字段只有：

- `open_trade_px`
- `high_trade_px`
- `low_trade_px`
- `close_trade_px`
- `vwap`
- `volume`
- `dvolume`
- `trade_count`

评估严格沿用项目内 score 逻辑：

`Score = (IC - 0.0005 * Turnover) * sqrt(IR) * 100`

并且没有在因子构造阶段使用 `resp` 或 `trading_restriction`，因此不存在未来信息泄漏。

## What Was Built

新增脚本：

- `manual/manual_factor_runner.py`

它负责：

- 一次性加载全量 15 分钟缓存并展开为宽表，加速批量评估
- 枚举手工设计的候选家族与参数
- 对每个候选同时测试正向 `pro` 和反向 `anti`
- 输出完整搜索结果与有效因子汇总

本次真实搜索结果已经保存为：

- `manual/artifacts/manual_factor_search_20260410_160914.csv`
- `manual/artifacts/manual_factor_effective_20260410_160914.csv`
- `manual/artifacts/manual_factor_family_summary_20260410_160914.csv`
- `manual/artifacts/manual_factor_selected_20260410_160914.csv`

## Search Summary

- 搜索候选数：66
- 过 gate 的有效因子数：26
- 有效家族数：16

整体结论非常清楚：

1. 最强的一组不是简单 VWAP 偏离，而是 bar 结构类因子，尤其是“收盘在本 bar 高低区间中的位置”。
2. 大多数有效因子都出现在 `pro` 方向，也就是短周期结构延续比单纯反转更有效。
3. 少数 `anti` 因子也很强，代表的是“更慢的平滑趋势差值”或“更长窗 stretch”的回归。
4. VWAP gap 相关因子虽然 IC 和 IR 很高，但因为仓位集中度超标而没法通过官方 gate，不能直接 submit。

## Strongest Effective Factors

下表是本次搜索里最值得汇报的有效因子：

| Rank | Key | Family | Dir | IC | IR | Turnover | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `range_location__base__pro` | Close Location In Range | pro | 0.0186 | 35.40 | 11.39 | 7.662 |
| 2 | `range_conditioned_location__window8__pro` | Range Location x Range Surprise | pro | 0.0171 | 31.59 | 11.05 | 6.491 |
| 3 | `wick_imbalance__base__pro` | Wick Imbalance | pro | 0.0146 | 37.67 | 11.22 | 5.522 |
| 4 | `zscore_body_fraction__window8__pro` | Body Fraction Z-Score | pro | 0.0141 | 45.89 | 12.31 | 5.348 |
| 5 | `ema_spread__long32_short8__anti` | Short-Long EMA Spread | anti | 0.0212 | 5.81 | 0.88 | 4.994 |
| 6 | `range_conditioned_location__window16__pro` | Range Location x Range Surprise | pro | 0.0153 | 24.34 | 10.95 | 4.842 |
| 7 | `zscore_body_fraction__window16__pro` | Body Fraction Z-Score | pro | 0.0140 | 35.63 | 12.24 | 4.704 |
| 8 | `ema_spread__long16_short4__anti` | Short-Long EMA Spread | anti | 0.0191 | 5.89 | 1.78 | 4.409 |
| 9 | `open_close_return__base__pro` | Open-Close Return | pro | 0.0135 | 20.51 | 11.71 | 3.445 |
| 10 | `range_conditioned_body__window16__pro` | Body x Range Surprise | pro | 0.0127 | 20.76 | 11.85 | 3.082 |

## What These Factors Mean

### 1. Range-location family is the strongest

最强因子 `range_location__base__pro` 的表达式可以理解为：

`rank((close - low) / (high - low + eps))`

含义很直接：

- 一根 15 分钟 bar 收盘越靠近本 bar 高点，因子值越高
- 这不是跨日趋势，而是 bar 内部成交结构的强弱
- 在这份数据上，它表现为显著的短周期延续，而不是回归

进一步把它与 `range_pct / rolling_mean(range_pct)` 相乘后，`range_conditioned_location` 仍然很强，说明：

- “收在高位”本身有效
- “而且这根 bar 的波动范围显著偏大”时，信号更纯

### 2. Wick/body family also works well

`wick_imbalance__base__pro` 和 `zscore_body_fraction__window8__pro` 说明：

- 下影线更强、上影线更弱，往往对应更强的后续表现
- 实体占比放大，并且相对自身历史分布变得异常时，信号更强

这类因子优点是：

- 可解释性强
- 只依赖单根 bar 及其短窗历史
- 和单纯 return 因子相比，包含了更细的 bar 结构信息

### 3. EMA spread is low-turnover alpha

`ema_spread__long32_short8__anti` 与 `ema_spread__long16_short4__anti` 的特点是：

- 分数不是靠超高 IR，而是靠极低换手
- `Turnover` 分别只有 `0.88` 和 `1.78`
- 在 score 公式下，这类 alpha 很值钱，因为 turnover 基本不拖后腿

它的含义是：

- 当短 EMA 相对长 EMA 明显抬升时，做反向排序
- 反映的是慢变量上的 stretch/reversion，而不是快变量上的 bar 结构延续

### 4. Return-conditioned liquidity family is a useful supplement

以下家族都通过了 gate，而且得分集中在 `2.5 ~ 2.8`：

- Return x Volume Surprise
- Return x Dollar-Volume Surprise
- Return x Trade-Count Surprise
- Return x Average Trade Value Surprise

这些因子不一定是绝对最强，但有两个优点：

- 因子定义直观，很适合汇报时解释成“量价共振”
- 家族内部参数稳定，说明不是偶然的单点 luck

## Family-Level Takeaways

按家族看，本次最值得保留的结论如下：

| Family | Best Signal | Interpretation |
| --- | --- | --- |
| Close Location In Range | `range_location__base__pro` | bar 内部收盘位置越强，后续越强 |
| Range Location x Range Surprise | `range_conditioned_location__window8__pro` | 强收盘 + 大范围 bar 更有效 |
| Wick Imbalance | `wick_imbalance__base__pro` | 下影支撑强于上影抛压时更有效 |
| Body Fraction Z-Score | `zscore_body_fraction__window8__pro` | 实体强度异常放大时有效 |
| Short-Long EMA Spread | `ema_spread__long32_short8__anti` | 慢变量 stretch 的反向排序有效且低换手 |
| Open-Close Return | `open_close_return__base__pro` | 同 bar 的 open-close 延续有效 |
| Body x Range Surprise | `range_conditioned_body__window16__pro` | 强实体在大 range bar 上更可信 |
| Close Z-Score Stretch | `close_zscore__window24__anti` | 更长窗的价格 stretch 偏向回归 |

## Why Some Seemingly Good Factors Failed

这部分很适合汇报时展示“我们不是只看 IC，还考虑 submit 可行性”。

### VWAP gap family

失败最典型的是：

- `vwap_gap__base__pro`
- `vwap_gap_with_dvol__window8__pro`
- `zscore_vwap_gap__window16__pro`

它们的问题不是 IC，而是集中度：

- `maxx = 5000` 或 `10000`
- `max_mean = 11.13` 或 `18.06`

也就是说：

- 因子排序会把仓位极端集中到极少数股票
- 尽管 IC/IR 看上去非常亮眼，也过不了官方 concentration gate
- 这类因子更像“研究线索”，不是当前版本能直接 submit 的形态

### Gap-only family

`gap_return__base__anti/pro` 效果都弱：

- IC 只有 `±0.0007`
- 说明单独的 open-vs-prev-close gap 在这份数据上没有足够稳定的信息量

### Too-short close z-score

`close_zscore__window8__anti` 没过 gate，但 `window24__anti` 能过：

- 过短窗口更容易吃到噪声
- 稍长 stretch 才更像真实可回归的状态变量

## Practical Reporting Narrative

如果要给导师或评委汇报，建议直接用下面这条主线：

1. 我们专门从高频 15 分钟 bar 的内部结构出发，手工设计了价量因子，而不是只依赖 LLM 自动生成。
2. 实测最有效的不是简单价格偏离，而是“bar 结构强弱”：
   典型就是 close 在 high-low 区间中的位置、上下影不对称、实体占比异常。
3. 数据表现出明显的短周期结构延续，而不是朴素反转。
4. 量价条件化因子整体稳定通过 gate，说明成交活跃度对信号有增强作用。
5. VWAP 偏离虽然有高 IC，但仓位过度集中，不适合直接 submit；这说明我们在筛选时把官方 gate 放在了第一位。
6. 此外还有一条很实用的低换手路线：
   EMA spread 的反向因子虽然不是最高 IC，但由于 turnover 极低，score 很不错。

## Current Recommendation

当前如果只保留最值得继续推进的少数家族，我建议优先级如下：

1. `range_location`
2. `range_conditioned_location`
3. `wick_imbalance`
4. `zscore_body_fraction`
5. `ema_spread` anti
6. `open_close_return`

如果下一轮继续做 manual 因子，可以优先沿这几个方向再扩展：

- 在 `range/location` 家族里加入 bar-of-day 或 session conditioning
- 在 `body/wick` 家族里引入 rolling stability 或 regime filter
- 在 `ema_spread` 家族里进一步搜索更长的 slow window，以继续压低 turnover

## Notes

- 本次“有效因子”判断以本地官方 gate 为准，而不是只看 IC。
- 大批量 parquet 导出在用户确认“不必凑够 20 个 submission”后被主动停下，以便优先产出可汇报结论。
- 如果需要把本轮 26 个有效因子再次批量导出为 submit 文件，可以在现有脚本基础上继续跑 export-only 流程，不需要重新搜索。
