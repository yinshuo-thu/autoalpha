# Manual 66 个测试因子设计说明

## 1. 这 66 个因子是怎么来的

这 66 个测试因子来自 [`manual/manual_factor_runner.py`](/Volumes/T7/autoalpha_v3/manual/manual_factor_runner.py) 里的 `generate_candidates()`。它不是 66 个完全无关的公式，而是按一套固定的手工研究框架批量展开出来的候选集合：

- 先定义 `20` 个 factor family。
- 每个 family 再按不同参数窗口展开。
- 每个参数版本同时测试两个方向：
  - `pro`：顺着原始信号方向做排序，偏“延续 / 跟随”解释。
  - `anti`：先把原始信号取反再排序，偏“反转 / 均值回归”解释。
- 所有候选在评估前都会统一转成截面 `rank`，保证输出形式一致，便于横向比较。

总数拆解如下：

- `close_zscore`：`3` 个窗口 x `2` 个方向 = `6`
- 8 个“单参数 family”：每个 `1` 套参数 x `2` 个方向 = `16`
- 11 个“多参数 family”：每个 `2` 套参数 x `2` 个方向 = `44`
- 合计：`6 + 16 + 44 = 66`

## 2. 设计边界与研究原则

这批 manual 因子有很明确的边界，重点是“简单、可解释、竞赛安全、便于系统扫描”。

- 只使用价量与成交行为字段：
  - `open_trade_px`
  - `high_trade_px`
  - `low_trade_px`
  - `close_trade_px`
  - `vwap`
  - `volume`
  - `dvolume`
  - `trade_count`
- 不把 `resp`、`trading_restriction` 一类标签或限制字段用于因子构造。
- 统一用短周期窗口做 15 分钟频率研究，主要覆盖 `8 / 15 / 16 / 24 / 32` bar。
- 每个思路都同时测“顺势”和“反向”，避免先验假设过强。

从设计思路上看，这 66 个候选主要是在回答五类问题：

1. 价格刚刚拉开以后，是继续走还是回吐？
2. K 线形态本身，能不能反映当下买卖力量？
3. 价格相对 VWAP 的偏离，是趋势信号还是均值回归信号？
4. 同样的涨跌，如果放在“放量 / 大单 / 高频成交”背景下，会不会更有信息量？
5. 把收益除以波动、把形态乘以 range surprise 之后，能否提高稳健性？

## 3. 分类总览

| 类别 | 包含 family | 数量 | 核心研究问题 |
| --- | --- | ---: | --- |
| 价格延续/反转类 | `close_zscore`、`open_close_return`、`gap_return`、`bar_return`、`ema_spread`、`multi_horizon_mix` | 18 | 短期价格拉伸、跳空和趋势到底是延续还是回归 |
| K 线形态类 | `range_location`、`body_fraction`、`wick_imbalance`、`zscore_body_fraction` | 10 | 单根 K 线内部结构是否反映即时强弱 |
| VWAP 偏离类 | `vwap_gap`、`vwap_gap_with_dvol`、`zscore_vwap_gap` | 10 | 收盘相对 VWAP 的偏离是否有持续性 |
| 成交活跃度条件化类 | `volume_conditioned_return`、`dvolume_conditioned_return`、`trade_conditioned_return`、`avg_trade_conditioned_return` | 16 | 同样的收益在不同成交背景下含义是否不同 |
| 波动/区间条件化类 | `volatility_conditioned_return`、`range_conditioned_body`、`range_conditioned_location` | 12 | 用波动和 range 做归一化后，信号是否更干净 |

## 4. 五大类因子的思路

### 4.1 价格延续/反转类

这类因子是最基础的一组，核心在于测试“刚发生的价格变化”本身是否有预测力。

- `close_zscore`
  - 用 `8 / 15 / 24` bar 滚动 z-score 衡量收盘价偏离均值的程度。
  - `pro` 假设强者恒强；`anti` 假设短线偏离会回归。
- `open_close_return`
  - 研究同一根 bar 内从开盘到收盘的涨跌是否有后续信息。
- `gap_return`
  - 研究本 bar 开盘相对前一 bar 收盘的跳空是否会继续扩散或回补。
- `bar_return`
  - 直接研究上一 bar 的 close-to-close return。
- `ema_spread`
  - 用短 EMA 相对长 EMA 的偏离来表达一个“非常轻量的 intraday 趋势斜率”。
  - 参数为 `4/16` 与 `8/32` 两组。
- `multi_horizon_mix`
  - 把 `1 / 4 / 16` bar 的收益叠加，测试“多时间尺度同向共振”是否更有效。

这一类的目标不是做复杂结构，而是先回答最基本的问题：纯价格信号在哪些窗口下更像趋势，在哪些窗口下更像反转。

### 4.2 K 线形态类

这类因子不强调跨 bar 趋势，而是利用单根 K 线的内部几何结构来表达多空力量。

- `range_location`
  - 看收盘在当根高低区间中的位置。
  - 收在高位可能代表买盘强，收在低位可能代表卖压强。
- `body_fraction`
  - 看实体长度相对整根 bar 振幅的占比。
  - 实体越大，通常表示方向性越明确。
- `wick_imbalance`
  - 比较上下影线不对称性。
  - 下影更长可能代表下方承接，上影更长可能代表上方抛压。
- `zscore_body_fraction`
  - 不是直接看实体大小，而是看实体强度相对最近 `8 / 16` bar 是否异常。

这一类的好处是可解释性很强，也容易和后续更复杂的 range/volatility 条件化思路做组合。

### 4.3 VWAP 偏离类

VWAP 是盘中很重要的“成交重心”。价格围绕 VWAP 的偏离，往往对应短线交易行为、冲击成本和均值回归压力。

- `vwap_gap`
  - 直接看 `close / vwap - 1`。
- `zscore_vwap_gap`
  - 不只看绝对偏离，还看这个偏离相对最近 `8 / 16` bar 是否异常。
- `vwap_gap_with_dvol`
  - 把 VWAP 偏离乘上 dollar-volume surprise，测试“高换手背景下的 VWAP 偏离”是否更有辨识度。

这一类本质是在测两种互相竞争的解释：

- 如果偏离来自持续主动买卖，`pro` 可能有效。
- 如果偏离来自短时冲击、尾盘挤压或流动性失衡，`anti` 可能更有效。

### 4.4 成交活跃度条件化类

这类因子是 manual 设计里最系统的一块。核心思想不是“收益本身”，而是“收益发生时伴随的交易活跃度结构”。

- `volume_conditioned_return`
  - 用成交量相对均值的放大量来加权 return。
- `dvolume_conditioned_return`
  - 用成交额 surprise 来加权 return。
- `trade_conditioned_return`
  - 用成交笔数 surprise 来加权 return。
- `avg_trade_conditioned_return`
  - 用平均单笔成交额 surprise 来加权 return。

它们分别想区分四种不同的市场状态：

- 放量但不一定大额成交。
- 金额显著放大，说明资金真实进出更强。
- 笔数放大，说明交易拥挤度和活跃度提高。
- 单笔均额变大，说明更可能是大单主导而非散单噪音。

这组 family 都用了 `8 / 16` 两个窗口，因为它们更像“短期背景状态”的比较，而不是超长趋势。

### 4.5 波动/区间条件化类

这类因子的目标是做“归一化”或“状态过滤”，避免把所有大涨大跌都当成同一种信号。

- `volatility_conditioned_return`
  - 用 `return / recent volatility` 表达收益相对当前波动背景的强弱。
  - 参数为 `16 / 32`。
- `range_conditioned_body`
  - 用 `body_frac * range surprise`，强调“实体大且 bar 振幅异常”的时刻。
- `range_conditioned_location`
  - 用 `range_loc * range surprise`，强调“收盘位置强且区间扩张异常”的时刻。

这组设计的直觉是：

- 在低波环境里，同样的 return 可能更有信息量。
- 在高 range 扩张时，K 线形态信号可能更“真实”，也可能更容易过冲。
- 所以必须同时保留 `pro` 和 `anti` 两个方向让数据自己说话。

## 5. 20 个 family 明细

| Family | 中文说明 | 候选数 | 参数展开 | 方向 | 研究意图 |
| --- | --- | ---: | --- | --- | --- |
| `close_zscore` | 收盘价滚动标准化偏离 | 6 | `window=8/15/24` | `pro/anti` | 测试价格拉伸是延续还是回归 |
| `range_location` | 收盘位于当根区间的位置 | 2 | `base` | `pro/anti` | 测试 close 靠近 high/low 是否有后续信息 |
| `body_fraction` | K 线实体占区间比例 | 2 | `base` | `pro/anti` | 测试实体强弱是否可预测 |
| `wick_imbalance` | 上下影线不平衡 | 2 | `base` | `pro/anti` | 测试承接/抛压结构 |
| `vwap_gap` | 收盘相对 VWAP 偏离 | 2 | `base` | `pro/anti` | 测试 VWAP 偏离的趋势或回归性 |
| `open_close_return` | 同 bar 开收到收盘收益 | 2 | `base` | `pro/anti` | 测试 bar 内方向性 |
| `gap_return` | 开盘相对前收跳空收益 | 2 | `base` | `pro/anti` | 测试跳空延续或回补 |
| `bar_return` | 相邻 bar close-to-close 收益 | 2 | `base` | `pro/anti` | 测试最短期 return 本身 |
| `volume_conditioned_return` | 收益 x 成交量 surprise | 4 | `window=8/16` | `pro/anti` | 测试放量背景下收益质量 |
| `dvolume_conditioned_return` | 收益 x 成交额 surprise | 4 | `window=8/16` | `pro/anti` | 测试真实资金活跃度 |
| `trade_conditioned_return` | 收益 x 笔数 surprise | 4 | `window=8/16` | `pro/anti` | 测试拥挤度和活跃度 |
| `avg_trade_conditioned_return` | 收益 x 平均单笔金额 surprise | 4 | `window=8/16` | `pro/anti` | 测试大单主导特征 |
| `volatility_conditioned_return` | 收益 / 近期波动率 | 4 | `window=16/32` | `pro/anti` | 做风险归一化 |
| `range_conditioned_body` | 实体强度 x 区间 surprise | 4 | `window=8/16` | `pro/anti` | 测试“强实体 + 异常波动” |
| `range_conditioned_location` | 收盘位置 x 区间 surprise | 4 | `window=8/16` | `pro/anti` | 测试“强收盘 + 区间扩张” |
| `vwap_gap_with_dvol` | VWAP 偏离 x 成交额 surprise | 4 | `window=8/16` | `pro/anti` | 测试偏离是否由真实换手驱动 |
| `ema_spread` | 短长 EMA 偏离 | 4 | `4/16`, `8/32` | `pro/anti` | 测试 intraday 趋势斜率 |
| `multi_horizon_mix` | 1/4/16 bar 收益混合 | 2 | `base` | `pro/anti` | 测试多周期共振 |
| `zscore_vwap_gap` | VWAP 偏离的滚动 z-score | 4 | `window=8/16` | `pro/anti` | 测试偏离是否“异常到值得交易” |
| `zscore_body_fraction` | 实体比例的滚动 z-score | 4 | `window=8/16` | `pro/anti` | 测试形态异常值 |

## 6. 为什么要同时保留 `pro` 和 `anti`

这 66 个候选里，最重要的不是公式复杂度，而是方向验证。

- 很多 intraday 信号在不同市场阶段会切换方向。
- 同样的价格拉伸，在趋势盘可能延续，在均值回归盘可能反转。
- 同样的 VWAP 偏离，在高流动性时可能是强势确认，在冲击性成交时可能是过冲。
- 同样的长下影，在某些股票代表承接，在另一些股票可能只是高波动噪音。

所以 manual 这批测试不是直接押方向，而是先把“思路”固定，再让 `pro/anti` 去回答方向问题。

## 7. 这 66 个候选的完整 key 清单

### 7.1 价格延续/反转类

`close_zscore`

- `close_zscore__window8__anti`
- `close_zscore__window8__pro`
- `close_zscore__window15__anti`
- `close_zscore__window15__pro`
- `close_zscore__window24__anti`
- `close_zscore__window24__pro`

`open_close_return`

- `open_close_return__base__anti`
- `open_close_return__base__pro`

`gap_return`

- `gap_return__base__anti`
- `gap_return__base__pro`

`bar_return`

- `bar_return__base__anti`
- `bar_return__base__pro`

`ema_spread`

- `ema_spread__long16_short4__anti`
- `ema_spread__long16_short4__pro`
- `ema_spread__long32_short8__anti`
- `ema_spread__long32_short8__pro`

`multi_horizon_mix`

- `multi_horizon_mix__base__anti`
- `multi_horizon_mix__base__pro`

### 7.2 K 线形态类

`range_location`

- `range_location__base__anti`
- `range_location__base__pro`

`body_fraction`

- `body_fraction__base__anti`
- `body_fraction__base__pro`

`wick_imbalance`

- `wick_imbalance__base__anti`
- `wick_imbalance__base__pro`

`zscore_body_fraction`

- `zscore_body_fraction__window8__anti`
- `zscore_body_fraction__window8__pro`
- `zscore_body_fraction__window16__anti`
- `zscore_body_fraction__window16__pro`

### 7.3 VWAP 偏离类

`vwap_gap`

- `vwap_gap__base__anti`
- `vwap_gap__base__pro`

`vwap_gap_with_dvol`

- `vwap_gap_with_dvol__window8__anti`
- `vwap_gap_with_dvol__window8__pro`
- `vwap_gap_with_dvol__window16__anti`
- `vwap_gap_with_dvol__window16__pro`

`zscore_vwap_gap`

- `zscore_vwap_gap__window8__anti`
- `zscore_vwap_gap__window8__pro`
- `zscore_vwap_gap__window16__anti`
- `zscore_vwap_gap__window16__pro`

### 7.4 成交活跃度条件化类

`volume_conditioned_return`

- `volume_conditioned_return__window8__anti`
- `volume_conditioned_return__window8__pro`
- `volume_conditioned_return__window16__anti`
- `volume_conditioned_return__window16__pro`

`dvolume_conditioned_return`

- `dvolume_conditioned_return__window8__anti`
- `dvolume_conditioned_return__window8__pro`
- `dvolume_conditioned_return__window16__anti`
- `dvolume_conditioned_return__window16__pro`

`trade_conditioned_return`

- `trade_conditioned_return__window8__anti`
- `trade_conditioned_return__window8__pro`
- `trade_conditioned_return__window16__anti`
- `trade_conditioned_return__window16__pro`

`avg_trade_conditioned_return`

- `avg_trade_conditioned_return__window8__anti`
- `avg_trade_conditioned_return__window8__pro`
- `avg_trade_conditioned_return__window16__anti`
- `avg_trade_conditioned_return__window16__pro`

### 7.5 波动/区间条件化类

`volatility_conditioned_return`

- `volatility_conditioned_return__window16__anti`
- `volatility_conditioned_return__window16__pro`
- `volatility_conditioned_return__window32__anti`
- `volatility_conditioned_return__window32__pro`

`range_conditioned_body`

- `range_conditioned_body__window8__anti`
- `range_conditioned_body__window8__pro`
- `range_conditioned_body__window16__anti`
- `range_conditioned_body__window16__pro`

`range_conditioned_location`

- `range_conditioned_location__window8__anti`
- `range_conditioned_location__window8__pro`
- `range_conditioned_location__window16__anti`
- `range_conditioned_location__window16__pro`

## 8. 文档用途建议

这份文档比较适合做三件事：

- 当作 manual 搜索的“因子地图”，快速知道每个 family 在测什么。
- 当作二轮实验的分组依据，比如只继续扩展 VWAP 类、只扩展成交条件化类。
- 当作结果复盘模板，后面可以把每类的胜率、通过率和最佳窗口补进来。

如果后面 manual 这轮跑完，我们还可以继续补一版“66 个候选的实测结果归纳”，把哪几类更强、哪些方向更稳定也一起总结进去。
