# bar 因子汇总

说明：
- 本文逐个整理 `factors.py` 中定义的因子函数，按源码出现顺序排列。
- `Formula` 采用源码级伪公式，尽量保留原始计算步骤；其中统一记 `N = 2^k`，`k` 来自 `period_id_list`。
- 为了便于阅读，`fillna(0)` 之类收尾步骤通常省略；零除保护、条件过滤等若影响含义，会在公式里保留。
- `含义` 是阅读参考，重点说明因子想刻画的市场特征，不等同于实际交易结论。

## fm2

Formula:
```text
N = 2^k, k in period_id_list
lowmin = ts_min(low, N)
cpl = ((close / lowmin) - 1)
cpl_sq_ma = ts_mean((cpl ** 2), N)
factor = (cpl_sq_ma ** 0.5)
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## fm2_2

Formula:
```text
N = 2^k, k in period_id_list
highmax = ts_max(high, N)
cph = ((close / highmax) - 1)
cph_sq_ma = ts_mean((cph ** 2), N)
factor = (cph_sq_ma ** 0.5)
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## fm2_3

Formula:
```text
N = 2^k, k in period_id_list
lowmin = ts_min(low, N)
highmax = ts_max(high, N)
mid = ((highmax + lowmin) / 2)
cpm = ((close / mid) - 1)
cpm_sq_ma = ts_mean((cpm ** 2), N)
factor = (cpm_sq_ma ** 0.5)
```
含义：关注收盘价在高低区间中的位置，以及价格区间本身的扩张或收缩。

## fm2_4

Formula:
```text
N = 2^k, k in period_id_list
mid = ((high + low) / 2)
cpl = ((vwap / mid) - 1)
cpl_sq_ma = ts_mean((cpl ** 2), N)
factor = (cpl_sq_ma ** 0.5)
```
含义：围绕成交均价重心，刻画价格相对成交重心的偏离与迁移。

## fm11

Formula:
```text
N = 2^k, k in period_id_list
amt_ma = ts_mean(amount, N)
amt_rank = ts_rank(amt_ma, N)
factor = amt_rank
```
含义：把当前观测放到历史窗口中排序，反映当前位置相对过去的高低分位。

## fm12

Formula:
```text
N = 2^k, k in period_id_list
vol_short = ts_mean(vol, N)
vol_long = ts_mean(vol, (N * 2))
vol_ratio = (vol_short / vol_long)
factor = vol_ratio
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm12_2

Formula:
```text
N = 2^k, k in period_id_list
vol_short = vol
vol_long = ts_mean(vol, N)
vol_ratio = (vol_short / vol_long)
factor = vol_ratio
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm14

Formula:
```text
N = 2^k, k in period_id_list
if N == 1:
    short_len = 1
else:
    short_len = (N / 2)
vol_short = ts_mean(vol, short_len)
vol_mean = ts_mean(vol_short, N)
vol_std = ts_std(vol_short, N)
vol_ratio = ((vol_short - vol_mean) / vol_std)
factor = vol_ratio
```
含义：关注波动率或离散度，衡量价格、收益或量能的稳定性。

## fm18

Formula:
```text
N = 2^k, k in period_id_list
clsma = ts_mean(close, N)
hl = (high - low)
factor = (hl / clsma)
```
含义：关注收盘价在高低区间中的位置，以及价格区间本身的扩张或收缩。

## fm21

Formula:
```text
N = 2^k, k in period_id_list
cl_mean = ts_mean(close, N)
cl_std = ts_std(close, N)
norm = ((close - cl_mean) / cl_std)
norm[abs(np) < 1e-05] = 0
factor = ts_argmax(norm, N)
```
含义：关注极值出现的时点或极值所在位置，衡量趋势拐点与极端样本的时间结构。

## fm21_2

Formula:
```text
N = 2^k, k in period_id_list
cl_mean = ts_mean(close, N)
cl_std = ts_std(close, N)
norm = ((close - cl_mean) / cl_std)
norm[abs(np) < 1e-05] = 0
factor = ts_argmin(norm, N)
```
含义：关注极值出现的时点或极值所在位置，衡量趋势拐点与极端样本的时间结构。

## fm21_3

Formula:
```text
N = 2^k, k in period_id_list
cl_mean = ts_mean(close, N)
cl_std = ts_std(close, N)
norm = ((close - cl_mean) / cl_std)
norm[abs(np) < 1e-05] = 0
factor = (ts_argmax(norm, N) - ts_argmin(norm, N))
```
含义：关注极值出现的时点或极值所在位置，衡量趋势拐点与极端样本的时间结构。

## fm25

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
vol_diff = delta(vol, 1)
sign_vr = (sign(vol_diff) * sign(ret))
factor = ts_mean(sign_vr, N)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm25_2

Formula:
```text
N = 2^k, k in period_id_list
sign_vr = (vol * 0)
factor = ts_mean(sign_vr, N)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm25_3

Formula:
```text
N = 2^k, k in period_id_list
sign_vr = (vol * 0)
factor = ts_mean(sign_vr, N)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm25_4

Formula:
```text
N = 2^k, k in period_id_list
sign_vr = (vol * 0)
factor = ts_mean(sign_vr, N)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm25_5

Formula:
```text
N = 2^k, k in period_id_list
sign_vr = (vol * 0)
factor = ts_mean(sign_vr, N)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm25_6

Formula:
```text
N = 2^k, k in period_id_list
sign_vr = (vol * 0)
factor = ts_mean(sign_vr, N)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm25_7

Formula:
```text
N = 2^k, k in period_id_list
sign_vr = (vol * 0)
factor = ts_mean(sign_vr, N)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm32

Formula:
```text
N = 2^k, k in period_id_list
hdis = (high - vwap)
ldis = (vwap - low)
hlrate = (hdis / ldis)
factor = ts_rank(hlrate, N)
```
含义：把当前观测放到历史窗口中排序，反映当前位置相对过去的高低分位。

## fm33

Formula:
```text
N = 2^k, k in period_id_list
hlrate = (high / low)
factor = ts_rank(hlrate, N)
```
含义：把当前观测放到历史窗口中排序，反映当前位置相对过去的高低分位。

## fm33_2

Formula:
```text
N = 2^k, k in period_id_list
hlrate = (high / low)
if N == 1:
    short_len = 1
else:
    short_len = (N / 2)
hl_ma = ts_mean(hlrate, (N / 2))
factor = ts_rank(hl_ma, N)
```
含义：把当前观测放到历史窗口中排序，反映当前位置相对过去的高低分位。

## fm34

Formula:
```text
N = 2^k, k in period_id_list
highmax = ts_max(high, N)
lowmin = ts_min(low, N)
mrm = (highmax / lowmin)
factor = ts_rank(mrm, N)
```
含义：把当前观测放到历史窗口中排序，反映当前位置相对过去的高低分位。

## fm49

Formula:
```text
N = 2^k, k in period_id_list
highmax = ts_max(high, N)
cph = ((close / highmax) - 1)
cphsq = (cph ** 2)
cphsq_ma = ts_mean(cphsq, N)
factor = (cphsq_ma ** 0.5)
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## fm61

Formula:
```text
N = 2^k, k in period_id_list
amt_ratio = (amount / ts_mean(amount, 48))
ret = ((close / delay(close, 1)) - 1)
if N == 1:
    short_len = 1
else:
    short_len = (N / 2)
ret_diff = (ts_mean(ret, short_len) - ts_mean(ret, N))
amount_std = ts_std(amt_ratio, N)
factor = (ret_diff / amount_std)
```
含义：关注波动率或离散度，衡量价格、收益或量能的稳定性。

## fm62

Formula:
```text
N = 2^k, k in period_id_list
amt_ratio = (amount / ts_mean(amount, 48))
ret = ((close / delay(close, 1)) - 1)
if N == 1:
    short_len = 1
else:
    short_len = (N / 2)
ret_diff = (ts_mean(ret, short_len) - ts_mean(ret, N))
amount_mean = ts_mean(amt_ratio, N)
factor = (ret_diff / amount_mean)
```
含义：围绕成交额变化，观察资金参与强度和放量/缩量特征。

## fm65

Formula:
```text
N = 2^k, k in period_id_list
hl = (high - low)
hlv = (hl * vol)
hlma = ts_mean(hl, N)
vma = ts_mean(vol, N)
hlvma1 = (hlma * vma)
hlvma2 = ts_mean(hlv, N)
factor = ((hlvma1 - hlvma2) / (hlvma1 + hlvma2))
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm66

Formula:
```text
N = 2^k, k in period_id_list
hl = (high + low)
hlv = (hl * vol)
hlma = ts_mean(hl, N)
vma = ts_mean(vol, N)
hlvma1 = (hlma * vma)
hlvma2 = ts_mean(hlv, N)
factor = ((hlvma1 - hlvma2) / (hlvma1 + hlvma2))
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm69

Formula:
```text
N = 2^k, k in period_id_list
hl_add = (high + low)
hl_diff = (high - low)
vol_ratio = (vol / ts_mean(vol, 48))
hlma = ts_mean(hl_diff, N)
hladdma = ts_mean(hl_add, N)
vma = ts_mean(vol_ratio, N)
mid = ((hl_add - hladdma) / hladdma)
factor = (((mid * vma) * hl_diff) / hlma)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm70

Formula:
```text
N = 2^k, k in period_id_list
hl_add = (high + low)
hl_diff = (high - low)
hlma = ts_mean(hl_diff, N)
hladdma = ts_mean(hl_add, N)
mid = ((hl_diff - hlma) / hlma)
mid[abs(np) < 1e-05] = 0
factor = ((mid * hl_add) / hladdma)
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## fm71

Formula:
```text
N = 2^k, k in period_id_list
hmax = ts_max(high, N)
vma = ts_mean(vol, N)
amtma = ts_mean(amount, N)
factor = ((amtma / vma) / hmax)
```
含义：同时结合成交额和成交量，刻画成交重心、流动性或量价配合关系。

## fm77

Formula:
```text
N = 2^k, k in period_id_list
clsma = ts_mean(close, N)
volma = ts_mean(vol, N)
cvma = ts_mean((close * vol), N)
factor = (cvma / (clsma * volma))
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm78

Formula:
```text
N = 2^k, k in period_id_list
volrank = ts_rank(vol, N)
factor = volrank
```
含义：把当前观测放到历史窗口中排序，反映当前位置相对过去的高低分位。

## fm79

Formula:
```text
N = 2^k, k in period_id_list
trend = (((close / low) + (high / open)) / 2.0)
trendmax = ts_max(trend, N)
factor = (trend / trendmax)
```
含义：关注收盘价在高低区间中的位置，以及价格区间本身的扩张或收缩。

## fm80

Formula:
```text
N = 2^k, k in period_id_list
trend = (((close / low) + (high / open)) / 2)
trend_max = ts_max(trend, N)
trend_min = ts_min(trend, N)
factor = ((trend_max - trend_min) / (trend_max + trend_min))
```
含义：关注收盘价在高低区间中的位置，以及价格区间本身的扩张或收缩。

## fm81

Formula:
```text
N = 2^k, k in period_id_list
trend = (((close / low) + (high / open)) / 2.0)
trendmin = ts_min(trend, N)
factor = (trend / trendmin)
```
含义：关注收盘价在高低区间中的位置，以及价格区间本身的扩张或收缩。

## fm82

Formula:
```text
N = 2^k, k in period_id_list
vwapr1 = ((vwap / delay(vwap, N)) ** (1 / N))
vwapr2 = ts_mean((vwap / delay(vwap, 1)), N)
factor = ((vwapr1 - vwapr2) * 10000)
```
含义：围绕成交均价重心，刻画价格相对成交重心的偏离与迁移。

## fm83

Formula:
```text
N = 2^k, k in period_id_list
hmax = ts_max(high, N)
lmin = ts_min(low, N)
cmax = ts_max(close, N)
cmin = ts_min(close, N)
pos1 = ((close - lmin) / (hmax - lmin))
pos2 = ((close - cmin) / (cmax - cmin))
pos1[abs(np) < 1e-05] = 0
pos2[abs(np) < 1e-05] = 0
factor = ((pos1 ** pos2) - (pos2 ** pos1))
```
含义：关注收盘价在高低区间中的位置，以及价格区间本身的扩张或收缩。

## fm87

Formula:
```text
N = 2^k, k in period_id_list
hmax = ts_max(high, N)
lmin = ts_min(low, N)
cmax = ts_max(close, N)
cmin = ts_min(close, N)
pos1 = ((close - lmin) / (hmax - lmin))
pos2 = ((close - cmin) / (cmax - cmin))
pos1[abs(np) < 1e-05] = 0
pos2[abs(np) < 1e-05] = 0
factor = ((pos1 - pos2) / (pos1 + pos2))
```
含义：关注收盘价在高低区间中的位置，以及价格区间本身的扩张或收缩。

## fm92

Formula:
```text
N = 2^k, k in period_id_list
factor = zeros(T)
vol = vol
factor[:] = 0
vwaprk = ts_rank(vwap, N)
for t in range(N, T):
    vol_wind = vol[((t - N) + 1):(t + 1)]
    vol_argmax = argmax(vol_wind)
    factor[t] = vwaprk[((t - N) + 1):(t + 1)][vol_argmax]
```
含义：关注极值出现的时点或极值所在位置，衡量趋势拐点与极端样本的时间结构。

## fm92_2

Formula:
```text
N = 2^k, k in period_id_list
factor = zeros(T)
if (N / 2) < 1:
    vol_len = 1
else:
    vol_len = (N / 2)
vol_array = ts_mean(vol, vol_len)
factor[:] = 0
vwaprk = ts_rank(vwap, N)
for t in range(N, T):
    vol_wind = vol_array[((t - N) + 1):(t + 1)]
    vol_argmax = argmax(vol_wind)
    factor[t] = vwaprk[((t - N) + 1):(t + 1)][vol_argmax]
```
含义：关注极值出现的时点或极值所在位置，衡量趋势拐点与极端样本的时间结构。

## fm93

Formula:
```text
N = 2^k, k in period_id_list
rawret = (close / delay(close, N))
vma = ts_mean(vol, N)
vrate = (vol / vma)
vrate[abs(np) < 1e-05] = 0
factor = (ts_std(vrate, N) * rawret)
```
含义：关注波动率或离散度，衡量价格、收益或量能的稳定性。

## fm94

Formula:
```text
N = 2^k, k in period_id_list
factor = zeros(T)
vol = vol
factor[:] = 0
volrk = ts_rank(vol, N)
for t in range(N, T):
    vwap_wind = vwap[((t - N) + 1):(t + 1)]
    vwap_argmax = argmax(vwap_wind)
    factor[t] = volrk[((t - N) + 1):(t + 1)][vwap_argmax]
```
含义：关注极值出现的时点或极值所在位置，衡量趋势拐点与极端样本的时间结构。

## fm102

Formula:
```text
N = 2^k, k in period_id_list
factor = (ts_argmax(amount, N) / N)
```
含义：关注极值出现的时点或极值所在位置，衡量趋势拐点与极端样本的时间结构。

## fm104

Formula:
```text
N = 2^k, k in period_id_list
factor = ((ts_argmax(close, N) / N) - (ts_argmax(amount, N) / N))
```
含义：关注极值出现的时点或极值所在位置，衡量趋势拐点与极端样本的时间结构。

## fm104_2

Formula:
```text
N = 2^k, k in period_id_list
factor = ((ts_argmin(close, N) / N) - (ts_argmax(amount, N) / N))
```
含义：关注极值出现的时点或极值所在位置，衡量趋势拐点与极端样本的时间结构。

## fm106

Formula:
```text
N = 2^k, k in period_id_list
a = zeros(len(vol))
b = zeros(len(vol))
vol_diff = delta(vol, 1)
a[:] = 0
b[:] = 0
a[vol_diff > 0] = 1
b[vol_diff < 0] = 1
abv = (a - b)
factor = (ts_mean(abv, N) / N)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm107

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
ret_abs = abs(ret)
ret_bak = (ret * 0)
ret_abs_quantile = ts_quantile(ret_abs, N, 0.5)
ret_bak[ret > ret_abs_quantile] = ret[ret > ret_abs_quantile]
factor = ts_mean(ret_bak, N)
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm107_2

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
ret_abs = abs(ret)
ret_bak = (ret * 0)
ret_abs_quantile = ts_quantile(ret_abs, N, 0.5)
ret_bak[ret < (-ret_abs_quantile)] = ret[ret < (-ret_abs_quantile)]
factor = ts_mean(ret_bak, N)
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm107_3

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
ret_abs = abs(ret)
ret_bak_down = (ret * 0)
ret_bak_up = (ret * 0)
ret_abs_quantile = ts_quantile(ret_abs, N, 0.5)
ret_bak_down[ret < (-ret_abs_quantile)] = ret[ret < (-ret_abs_quantile)]
ret_bak_up[ret > ret_abs_quantile] = ret[ret > ret_abs_quantile]
factor = (ts_mean(ret_bak_up, N) + ts_mean(ret_bak_down, N))
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm108

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
ret_bak = (ret * 0)
vol_quantile = ts_quantile(vol, N, 0.5)
ret_bak[vol > vol_quantile] = ret[vol > vol_quantile]
factor = ts_mean(ret_bak, N)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm110

Formula:
```text
N = 2^k, k in period_id_list
deltal = (-delta(open, 1))
mdm = (deltal * 0)
deltah = delta(close, 1)
pdm = (deltah * 0)
pdm_ma = ts_mean(pdm, N)
mdm_ma = ts_mean(mdm, N)
fenzi = (pdm_ma + mdm_ma)
factor = ((pdm_ma - mdm_ma) / fenzi)
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## fm112

Formula:
```text
N = 2^k, k in period_id_list
adv = (0 * open)
advvol = (adv * vol)
dec = (0 * open)
decvol = (dec * vol)
advsum = ts_sum(adv, N)
decsum = ts_sum(dec, N)
advvolsum = ts_sum(advvol, N)
decvolsum = ts_sum(decvol, N)
fenzi = (decsum * advvolsum)
factor = ((advsum * decvolsum) / fenzi)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm113

Formula:
```text
N = 2^k, k in period_id_list
adv = (0 * open)
ret = ((close / delay(close, 1)) - 1)
advret = (adv * ret)
dec = (0 * open)
decret = (dec * ret)
advsum = ts_sum(adv, N)
decsum = ts_sum(dec, N)
advretsum = ts_sum(advret, N)
decretsum = ts_sum(decret, N)
fenzi = (decsum * advretsum)
factor = ((advsum * decretsum) / fenzi)
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm114

Formula:
```text
N = 2^k, k in period_id_list
vwap_ret = ((vwap / delay(vwap, N)) - 1)
vpt = (vwap_ret * vol)
factor = ts_rank(vpt, N)
```
含义：把当前观测放到历史窗口中排序，反映当前位置相对过去的高低分位。

## fm114_2

Formula:
```text
N = 2^k, k in period_id_list
vwap_ret = ((vwap / delay(vwap, N)) - 1)
vpt = vwap_ret
factor = ts_rank(vpt, N)
```
含义：把当前观测放到历史窗口中排序，反映当前位置相对过去的高低分位。

## fm116

Formula:
```text
N = 2^k, k in period_id_list
down = (open * 0)
up = (open * 0)
up_sum = ts_mean(up, N)
down_sum = ts_mean(down, N)
fenzi = (up_sum + down_sum)
factor = ((up_sum - down_sum) / fenzi)
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## fm117

Formula:
```text
N = 2^k, k in period_id_list
hpc = (high - delay(close, 1))
pcl = (delay(close, 1) - low)
hpc_mean = ts_mean(hpc, N)
pcl_mean = ts_mean(pcl, N)
fenzi = pcl_mean
factor = (hpc_mean / fenzi)
```
含义：关注收盘价在高低区间中的位置，以及价格区间本身的扩张或收缩。

## fm118

Formula:
```text
N = 2^k, k in period_id_list
premid = delay(((high + low) / 2), 1)
hpc = (high - premid)
pcl = (premid - low)
hpc_mean = ts_mean(hpc, N)
pcl_mean = ts_mean(pcl, N)
fenzi = pcl_mean
factor = (hpc_mean / fenzi)
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## fm122

Formula:
```text
N = 2^k, k in period_id_list
pre_vwap = delay(vwap, 1)
sizeup = (0 * open)
sizedown = (0 * open)
numup = (0 * open)
numdown = (0 * open)
ret = ((close - open) / open)
sizeup[vwap > pre_vwap] = ret[vwap > pre_vwap]
sizedown[vwap < pre_vwap] = (-ret[vwap < pre_vwap])
numup[vwap > pre_vwap] = 1
numdown[vwap < pre_vwap] = 1
sizeup_sum = ts_sum(sizeup, N)
numup_sum = ts_sum(numup, N)
sizedown_sum = ts_sum(sizedown, N)
numdown_sum = ts_sum(numdown, N)
qbang = (((sizeup_sum * numup_sum) - (sizedown_sum * numdown_sum)) / ((sizeup_sum * numup_sum) + (sizedown_sum * numdown_sum)))
fenmu = ((sizeup_sum * numup_sum) + (sizedown_sum * numdown_sum))
qbang[abs(fenmu) < 1e-09] = 0
factor = qbang
```
含义：围绕成交均价重心，刻画价格相对成交重心的偏离与迁移。

## fm124

Formula:
```text
N = 2^k, k in period_id_list
pre_vwap = delay(vwap, 1)
sizeup = (0 * open)
sizedown = (0 * open)
ret = ((vwap - pre_vwap) / pre_vwap)
sizeup[vwap > pre_vwap] = ret[vwap > pre_vwap]
sizedown[vwap < pre_vwap] = (-ret[vwap < pre_vwap])
sizeup_sum = ts_sum(sizeup, N)
sizedown_sum = ts_sum(sizedown, N)
qbang = ((sizeup_sum - sizedown_sum) / (sizeup_sum + sizedown_sum))
fenmu = (sizeup_sum + sizedown_sum)
qbang[abs(fenmu) < 1e-09] = 0
factor = qbang
```
含义：围绕成交均价重心，刻画价格相对成交重心的偏离与迁移。

## fm128

Formula:
```text
N = 2^k, k in period_id_list
add = (0 * open)
minus = (0 * open)
tvi = (add - minus)
factor = ts_mean(tvi, N)
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## fm128_2

Formula:
```text
N = 2^k, k in period_id_list
add = (0 * open)
minus = (0 * open)
tvi = ((add - minus) / (add + minus))
factor = ts_mean(tvi, N)
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## fm128_3

Formula:
```text
N = 2^k, k in period_id_list
add = (0 * open)
minus = (0 * open)
tvi = ((add - minus) / (add + minus))
factor = ts_mean(tvi, N)
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## fm128_4

Formula:
```text
N = 2^k, k in period_id_list
add = (0 * open)
minus = (0 * open)
tvi = ((add - minus) / (add + minus))
factor = ts_mean(tvi, N)
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## fm129

Formula:
```text
N = 2^k, k in period_id_list
hl = (high - low)
vhl = (((2 * vwap) - high) - low)
ama = ts_mean(vhl, N)
bma = ts_mean(hl, N)
factor = (ama / bma)
```
含义：围绕成交均价重心，刻画价格相对成交重心的偏离与迁移。

## fm132

Formula:
```text
N = 2^k, k in period_id_list
delta = (0 * vwap)
sign_vol = (delta * vol)
ado = ((sign_vol * (vwap - low)) / (high - low))
factor = ts_mean(ado, N)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm133

Formula:
```text
N = 2^k, k in period_id_list
delta = (0 * vwap)
sign_vol = (delta * vol)
ado = ((sign_vol * (close - low)) / (high - low))
factor = ts_mean(ado, N)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm136

Formula:
```text
N = 2^k, k in period_id_list
tmpval = ((vol * (((2 * close) - low) - high)) / (high - low))
tmpsum = ts_mean(tmpval, N)
volsum = ts_mean(vol, N)
factor = (tmpsum / volsum)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm136_2

Formula:
```text
N = 2^k, k in period_id_list
tmpval = ((((2 * close) - low) - high) / (high - low))
tmpma = ts_mean(tmpval, N)
factor = tmpma
```
含义：关注收盘价在高低区间中的位置，以及价格区间本身的扩张或收缩。

## fm137

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
volret = ((vol / delay(vol, 1)) - 1)
sumret = ts_sum(ret, N)
sumvolret = ts_sum(volret, N)
factor = (sumret * sumvolret)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm138

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
volret = ((vol / delay(vol, 1)) - 1)
dbret = (ret * volret)
factor = ts_mean(dbret, N)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm139

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
volret = ((vol / delay(vol, 1)) - 1)
dbret = (ret + volret)
factor = ts_mean(dbret, N)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm141

Formula:
```text
N = 2^k, k in period_id_list
dn = (open * 0)
un = (open * 0)
udn = (un + dn)
if N == 1:
    short_len = 1
else:
    short_len = (N / 2)
sma = ts_mean(udn, short_len)
lma = ts_mean(udn, N)
factor = (sma - lma)
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## fm143

Formula:
```text
N = 2^k, k in period_id_list
homean = ts_mean((high - open), N)
olmean = ts_mean((open - low), N)
factor = (homean / olmean)
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## fm144

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
factor = ts_max(ret, N)
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm144_2

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
if N == 1:
    period_len = 1
else:
    period_len = (N / 2)
ret_ma = ts_mean(ret, period_len)
factor = ts_max(ret_ma, N)
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm144_3

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
if N == 1:
    period_len = 1
else:
    period_len = (N / 2)
ret_ma = ts_mean(ret, period_len)
factor = ts_min(ret_ma, N)
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm144_4

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
if N == 1:
    period_len = 1
else:
    period_len = (N / 2)
ret_ma = ts_mean(ret, period_len)
factor = (ts_max(ret_ma, N) - ts_min(ret_ma, N))
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm144_5

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
factor = ts_min(ret, N)
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm144_6

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
factor = (ts_max(ret, N) - ts_min(ret, N))
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm145

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
abs_ret = abs(ret)
ret_ma = ts_mean(ret, N)
abs_ret_ma = ts_mean(abs_ret, N)
factor = (ret_ma / abs_ret_ma)
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm146

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
sign = sign(ret)
sign_amount = (sign * amount)
signamount_ma = ts_mean(sign_amount, N)
amount_ma = ts_mean(amount, N)
factor = (signamount_ma / amount_ma)
```
含义：围绕成交额变化，观察资金参与强度和放量/缩量特征。

## fm150

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
ret_vol = (abs(np) / sqrt(vol))
vwap_bak = (0 * vwap)
vwapma = ts_mean(vwap, N)
retvol_quantile = ts_quantile(ret_vol, N, 0.5)
vwap_bak[ret_vol > retvol_quantile] = vwap[ret_vol > retvol_quantile]
factor = (ts_mean(vwap_bak, N) / vwapma)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm161

Formula:
```text
N = 2^k, k in period_id_list
vwap_bak = (0 * vwap)
vwapma = ts_mean(vwap, N)
amount_quantile = ts_quantile(amount, N, 0.8)
vwap_bak[amount > amount_quantile] = vwap[amount > amount_quantile]
factor = (ts_mean(vwap_bak, N) / vwapma)
```
含义：围绕成交额变化，观察资金参与强度和放量/缩量特征。

## fm162

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
ret_vol = (abs(np) / amount)
factor = ts_rank(ret_vol, N)
```
含义：把当前观测放到历史窗口中排序，反映当前位置相对过去的高低分位。

## fm163

Formula:
```text
N = 2^k, k in period_id_list
ret_dw = zeros(T)
vol_dw = zeros(T)
illq_dw = (abs(ret_dw) / (sqrt(vol_dw) + 0.0001))
ret_up = zeros(T)
vol_up = zeros(T)
illq_up = (abs(ret_up) / (sqrt(vol_up) + 0.0001))
factor = (ts_mean(illq_up, N) - ts_mean(illq_dw, N))
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm164

Formula:
```text
N = 2^k, k in period_id_list
ret_dw = zeros(T)
ret_dw_2 = (ret_dw ** 2)
ret_up = zeros(T)
ret_up_2 = (ret_up ** 2)
factor = (ts_mean(ret_up_2, N) - ts_mean(ret_dw_2, N))
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm165

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
pvt = (ret * vol)
pvt_sum = ts_sum(pvt, N)
factor = ts_rank(pvt_sum, N)
```
含义：把当前观测放到历史窗口中排序，反映当前位置相对过去的高低分位。

## fm172

Formula:
```text
N = 2^k, k in period_id_list
vol_ma = ts_mean(vol, N)
vol_std_short = ts_std(vol_ma, N)
vol_std_long = ts_std(vol_ma, (N * 2))
factor = (vol_std_short / vol_std_long)
```
含义：关注波动率或离散度，衡量价格、收益或量能的稳定性。

## fm173

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
ret_p2 = (ret ** 2)
ret_p3 = (ret ** 3)
ret_p2_ma = ts_mean(ret_p2, N)
ret_p3_ma = ts_mean(ret_p3, N)
factor = (ret_p3_ma / (ret_p2_ma ** 1.5))
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm174

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
ret_p2 = (ret ** 2)
ret_p3 = (ret ** 4)
ret_p2_ma = ts_mean(ret_p2, N)
ret_p3_ma = ts_mean(ret_p3, N)
factor = (ret_p3_ma / (ret_p2_ma ** 2))
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm179

Formula:
```text
N = 2^k, k in period_id_list
vpct = (vol / ts_mean(vol, N))
factor = ts_mean(vpct, N)
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm181

Formula:
```text
N = 2^k, k in period_id_list
factor = zeros(T)
factor[:] = 0
for t in range(N, T):
    vol_wind = vol[((t - N) + 1):(t + 1)]
    cls_wind = close[((t - N) + 1):(t + 1)]
    cls_last = cls_wind[(-1)]
    factor[t] = (sum(vol_wind[cls_wind > cls_last]) / sum(vol_wind))
```
含义：围绕成交量结构，观察量能扩张、缩量或量价配合的特征。

## fm184

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
sumret = ts_sum(ret, N)
factor = ts_max(sumret, N)
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm184_2

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
sumret = ts_sum(ret, N)
factor = ts_min(sumret, N)
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm184_3

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
sumret = ts_sum(ret, N)
factor = (ts_max(sumret, N) - ts_min(sumret, N))
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm185

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
neg = (0 * ret)
pos = (0 * ret)
factor = (ts_mean(pos, N) - ts_mean(neg, N))
```
含义：围绕收益率、动量或反转构造，刻画价格延续性与回撤特征。

## fm187

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
ret_bak = (0 * ret)
amount_percentile = ts_quantile(amount, N, 0.8)
ret_bak[amount > amount_percentile] = ret[amount > amount_percentile]
factor = ts_mean(ret_bak, N)
```
含义：围绕成交额变化，观察资金参与强度和放量/缩量特征。

## fm187_2

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
ret_bak = (0 * ret)
amount_percentile = ts_quantile(amount, N, 0.2)
ret_bak[amount < amount_percentile] = ret[amount < amount_percentile]
factor = ts_mean(ret_bak, N)
```
含义：围绕成交额变化，观察资金参与强度和放量/缩量特征。

## fm187_3

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
ret_bak_20 = (0 * ret)
ret_bak_80 = (0 * ret)
amount_percentile_80 = ts_quantile(amount, N, 0.8)
ret_bak_80[amount > amount_percentile_80] = ret[amount > amount_percentile_80]
amount_percentile_20 = ts_quantile(amount, N, 0.2)
ret_bak_20[amount < amount_percentile_20] = ret[amount < amount_percentile_20]
factor = (ts_mean(ret_bak_80, N) - ts_mean(ret_bak_20, N))
```
含义：围绕成交额变化，观察资金参与强度和放量/缩量特征。

## fm188

Formula:
```text
N = 2^k, k in period_id_list
diff = zeros(T)
acd = ts_mean(diff, N)
clsma = ts_mean(close, N)
factor = (acd / clsma)
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## fm189

Formula:
```text
N = 2^k, k in period_id_list
dbm = dbm
dtm = dtm
stm = ts_sum(dtm, N)
sbm = ts_sum(dbm, N)
r1 = ((stm - sbm) / stm)
r2 = ((stm - sbm) / sbm)
factor = r1
factor[stm < sbm] = r2[stm < sbm]
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## fm190

Formula:
```text
N = 2^k, k in period_id_list
dbm = dbm
dtm = dtm
stm = ts_sum(dtm, N)
sbm = ts_sum(dbm, N)
r1 = ((stm - sbm) / stm)
r2 = ((stm - sbm) / sbm)
factor = r1
factor[stm < sbm] = r2[stm < sbm]
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## fm191

Formula:
```text
N = 2^k, k in period_id_list
pre_cl = delay(close, 1)
hmpc = (high - pre_cl)
max1 = (hmpc * 0)
pcml = (pre_cl - low)
max2 = (pcml * 0)
denom = ts_sum(max1, N)
numer = ts_sum(max2, N)
factor = (denom / numer)
```
含义：关注收盘价在高低区间中的位置，以及价格区间本身的扩张或收缩。

## fm192

Formula:
```text
N = 2^k, k in period_id_list
hl = (high / low)
factor = ts_mean(hl, N)
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## dev

Formula:
```text
N = 2^k, k in period_id_list
factor = ((close - ts_mean(close, N)) / ts_mean(close, N))
```
含义：从标准 bar 数据中提取价格、波动、量能或区间结构特征，用来描述当前市场状态。

## rawret

Formula:
```text
N = 2^k, k in period_id_list
factor = ts_sum(cp_ret, N)
```
含义：累计原始收益率，直接刻画一段窗口内的价格动量强弱。

## rwi

Formula:
```text
N = 2^k, k in period_id_list
factor = rwi
```
含义：Random Walk Index 风格指标，用真实波幅归一化价格向上/向下偏离，衡量趋势是否强于随机游走。

## dpo

Formula:
```text
N = 2^k, k in period_id_list
ma_short = ts_mean(close, N)
ma_long = ts_mean(close, (N * 2))
factor = ((ma_short - ma_long) / ma_long)
```
含义：短均线与长均线的相对偏离，反映价格相对趋势基线的高低位置。

## m9

Formula:
```text
N = 2^k, k in period_id_list
highmax = ts_max(high, N)
lowmin = ts_min(low, N)
mid = ((highmax + lowmin) / 2)
vwapmean = ts_mean(vwap, N)
m9 = ((vwapmean - mid) / vwapmean)
m9[abs(vwapmean) < 1e-09] = 0
factor = m9
```
含义：VWAP 均值相对区间中位的偏离，刻画成交重心是否偏向区间上沿或下沿。

## cv

Formula:
```text
N = 2^k, k in period_id_list
factor = cv
```
含义：价量乘积均值与均价均量乘积的比值，描述价格与成交量的协同程度。

## cr

Formula:
```text
N = 2^k, k in period_id_list
premid = ((delay(high, 1) + delay(low, 1)) / 2)
hm = (high - premid)
ml = (premid - low)
hmsum = ts_sum(hm, N)
mlsum = ts_sum(ml, N)
cr = (hmsum / mlsum)
cr[abs(mlsum) < 1e-05] = 0
factor = cr
```
含义：CR 指标风格的强弱比，衡量价格对前一周期中枢的上攻与下探力量。

## rsi

Formula:
```text
N = 2^k, k in period_id_list
factor = rsi
```
含义：RSI 风格的相对强弱指标，比较上涨幅度与总波动幅度。

## cmo

Formula:
```text
N = 2^k, k in period_id_list
diff = delta(close, 1)
su = (close * 0)
sd = (close * 0)
su[close > delay(close, 1)] = diff[close > delay(close, 1)]
sd[close < delay(close, 1)] = (-diff[close < delay(close, 1)])
su_sum = ts_sum(su, N)
sd_sum = ts_sum(sd, N)
cmo = ((su_sum - sd_sum) / (su_sum + sd_sum))
fenmu = (su_sum + sd_sum)
cmo[abs(fenmu) < 1e-09] = 0
factor = cmo
```
含义：Chande Momentum Oscillator，衡量上涨动能与下跌动能的相对强弱。

## whlr

Formula:
```text
N = 2^k, k in period_id_list
factor = (((hh / prell) + (ll / prehh)) / 2)
```
含义：当前窗口高低点相对前序窗口高低点的位置，刻画突破与回撤结构。

## bop

Formula:
```text
N = 2^k, k in period_id_list
co = (close - open)
hl = (high - low)
co_hl = (co / hl)
factor = ts_mean(co_hl, N)
```
含义：Balance of Power，比较收盘相对开盘及区间振幅的位置，衡量多空谁更占优。

## real

Formula:
```text
N = 2^k, k in period_id_list
highmax = ts_max(high, N)
lowmin = ts_min(low, N)
ret = (close - delay(close, N))
temp = (ret / (highmax - lowmin))
temp[(highmax - lowmin) < 1e-05] = 0
factor = temp
```
含义：实体长度相对波动区间或均值的位置，刻画 K 线实体强弱。

## dis

Formula:
```text
N = 2^k, k in period_id_list
highmax = ts_max(high, N)
lowmin = ts_min(low, N)
mid = ((highmax + lowmin) / 2)
temp = ((close - mid) / (highmax - lowmin))
temp[(highmax - lowmin) < 1e-05] = 0
factor = temp
```
含义：价格相对均线或参考价格的偏离程度，反映短期超涨超跌。

## sharp

Formula:
```text
N = 2^k, k in period_id_list
factor = sharp
```
含义：Sharpe 风格指标，用均值/波动率衡量收益的风险调整后强度。

## ddi

Formula:
```text
N = 2^k, k in period_id_list
dmf = (0 * high)
dmz = (0 * high)
diz = ts_sum(dmz, N)
dif = ts_sum(dmf, N)
fenmu = (diz + dif)
factor = ((diz - dif) / fenmu)
```
含义：方向性波动差异指标，刻画上行与下行波动哪个更占主导。

## auad

Formula:
```text
N = 2^k, k in period_id_list
high = high
low = low
h_dis = ((N - 1) - ts_argmax(high, N))
l_dis = ((N - 1) - ts_argmin(low, N))
au = ((N - h_dis) / N)
ad = ((N - l_dis) / N)
auad = ((au - ad) / (au + ad))
factor = auad
```
含义：涨跌推进类指标，比较上涨样本和下跌样本的累计贡献。

## qbang

Formula:
```text
N = 2^k, k in period_id_list
sizeup = (0 * open)
sizedown = (0 * open)
numup = (0 * open)
numdown = (0 * open)
ret = ((close - open) / open)
sizeup[close > open] = ret[close > open]
sizedown[close < open] = (-ret[close < open])
numup[close > open] = 1
numdown[close < open] = 1
sizeup_sum = ts_sum(sizeup, N)
numup_sum = ts_sum(numup, N)
sizedown_sum = ts_sum(sizedown, N)
numdown_sum = ts_sum(numdown, N)
qbang = (((sizeup_sum * numup_sum) - (sizedown_sum * numdown_sum)) / ((sizeup_sum * numup_sum) + (sizedown_sum * numdown_sum)))
fenmu = ((sizeup_sum * numup_sum) + (sizedown_sum * numdown_sum))
qbang[abs(fenmu) < 1e-09] = 0
factor = qbang
```
含义：区间突破/强弱风格指标，衡量价格在历史区间中的攻击性。

## rank

Formula:
```text
N = 2^k, k in period_id_list
close = close
close_rank = ts_rank(close, N)
factor = close_rank
```
含义：把原始序列映射到窗口历史分位，反映当前位置相对历史高低。

## hhv

Formula:
```text
N = 2^k, k in period_id_list
hh = ts_max(dh, N)
dv1 = (ts_mean((dvwap * dv), N) / ts_mean(dv, N))
hhv = (dv1 / hh)
hhv[abs(hh) < 1e-09] = 0
hhv[abs(hhv) > 1000] = 0
factor = hhv
```
含义：历史高点位置指标，刻画当前价格离近期最高水平有多近。

## tii

Formula:
```text
N = 2^k, k in period_id_list
close_ma = ts_mean(close, N)
dev = (close - close_ma)
dev_pos = (0 * close)
dev_neg = (0 * close)
dev_pos[dev > 0] = dev
dev_neg[dev < 0] = (-dev)
sum_pos = ts_sum(dev_pos, N)
sum_neg = ts_sum(dev_neg, N)
fenmu = (sum_pos + sum_neg)
factor = (sum_pos / fenmu)
```
含义：Trend Intensity Index 风格指标，衡量价格相对趋势中枢的一致性。

## dmi

Formula:
```text
N = 2^k, k in period_id_list
mdm = (-delta(low, 1))
pdm = delta(high, 1)
pdi = ts_sum(pdm, N)
mdi = ts_sum(mdm, N)
fenmu = (pdi + mdi)
dmi = ((pdi - mdi) / (pdi + mdi))
dmi[abs(fenmu) < 1e-05] = 0
factor = dmi
```
含义：Directional Movement Index 风格指标，比较正向与反向价格推进力度。

## wvad

Formula:
```text
N = 2^k, k in period_id_list
a = (close - open)
b = (high - low)
ab = (a / b)
wvad = (ts_mean((ab * vol), N) / ts_mean(vol, N))
fenmu = ts_mean(vol, N)
wvad[abs(fenmu) < 1e-09] = 0
factor = wvad
```
含义：Williams Variable Accumulation/Distribution，结合涨跌方向与成交量衡量累积买卖压。

## macd

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
dif = (ts_mean(ret, N) - ts_mean(ret, (2 * N)))
dea = ts_mean(dif, N)
macd = (dif - dea)
factor = macd
```
含义：MACD 风格指标，比较快慢均线差异，衡量趋势强弱与拐点。

## adx

Formula:
```text
N = 2^k, k in period_id_list
hl = (high - low)
tr = hl
tr = ts_sum(tr, N)
factor = adx
```
含义：Average Directional Index 风格指标，衡量趋势方向强度而非涨跌方向本身。

## vhf

Formula:
```text
N = 2^k, k in period_id_list
highmax = ts_max(high, N)
lowmin = ts_min(low, N)
diff_abs = abs(delta(close, 1))
fenmu = ts_sum(diff_abs, N)
vhf = ((highmax - lowmin) / fenmu)
vhf[abs(fenmu) < 1e-09] = 0
factor = vhf
```
含义：Vertical Horizontal Filter，比较净位移与路径长度，判断趋势性还是震荡性。

## mfi

Formula:
```text
N = 2^k, k in period_id_list
factor = mfi
```
含义：Money Flow Index 风格指标，结合典型价格与成交量刻画资金流强弱。

## kdjd

Formula:
```text
N = 2^k, k in period_id_list
highmax = ts_max(high, N)
lowmin = ts_min(low, N)
fenmu = (highmax - lowmin)
stoch = ((close - lowmin) / (highmax - lowmin))
stoch[abs(fenmu) < 1e-05] = 0
stoch_max = ts_max(stoch, N)
stoch_min = ts_min(stoch, N)
fenmu = (stoch_max - stoch_min)
factor = ((stoch - stoch_min) / fenmu)
```
含义：KDJ/D 指标风格，刻画价格在区间中的相对位置及平滑后的动量。

## pdi

Formula:
```text
N = 2^k, k in period_id_list
factor = pdi
```
含义：正向动量强度指标，衡量向上推进的幅度占比。

## boll

Formula:
```text
N = 2^k, k in period_id_list
close_ma = ts_mean(close, N)
close_std = ts_std(close, N)
boll = ((close - close_ma) / close_std)
boll[abs(close_std) < 1e-09] = 0
factor = boll
```
含义：布林带标准分数，衡量价格偏离滚动均值的标准差距离。

## adtm

Formula:
```text
N = 2^k, k in period_id_list
dbm = (0 * open)
dtm = (0 * open)
stm = ts_sum(dtm, N)
sbm = ts_sum(dbm, N)
fenmu = (stm + sbm)
factor = ((stm - sbm) / fenmu)
```
含义：ADTM 风格指标，比较开盘后向上扩张与向下扩张的力量差。

## abi

Formula:
```text
N = 2^k, k in period_id_list
a_boll = (close * 0)
b_boll = (close * 0)
a_boll[close > open] = 1
b_boll[close < open] = 1
a = ts_sum(a_boll, N)
b = ts_sum(b_boll, N)
abi = ((a - b) / (a + b))
fenmu = (a + b)
abi[abs(fenmu) < 1e-05] = 0
factor = abi
```
含义：上涨 K 线数量与下跌 K 线数量的相对差值，反映多空天数占优关系。

## obv

Formula:
```text
N = 2^k, k in period_id_list
dire_vol = vol
temp = (ts_mean(dire_vol, N) / ts_mean(vol, N))
temp[ts_mean(vol, N) < 1e-09] = 0
factor = temp
```
含义：On-Balance Volume 风格指标，把量能按涨跌方向签名后累积，衡量量价同向性。

## up

Formula:
```text
N = 2^k, k in period_id_list
factor = ts_mean(up_num, N)
```
含义：上涨比例，衡量窗口内收盘上涨样本占比。

## adjsharp

Formula:
```text
N = 2^k, k in period_id_list
ret_down = (0 * close)
ret_up = (0 * close)
up_mean = ts_mean(ret_up, N)
up_std = ts_std(ret_up, N)
down_mean = ts_mean(ret_down, N)
down_std = ts_std(ret_down, N)
up_sharp = (up_mean / up_std)
up_sharp[abs(np) < 1e-10] = 0
down_sharp = (down_mean / down_std)
down_sharp[abs(np) < 1e-10] = 0
factor = (up_sharp + down_sharp)
```
含义：分别考察上涨收益和下跌收益的 Sharpe，再合成整体风险调整收益。

## cjb

Formula:
```text
N = 2^k, k in period_id_list
cjb = ((close / ts_min(low, N)) - (close / delay(close, N)))
factor = cjb
```
含义：当前收盘相对窗口低点与滞后收益的比较，衡量从低位反弹的力度。

## htb

Formula:
```text
N = 2^k, k in period_id_list
htb = ((close / ts_max(high, N)) - (close / delay(close, N)))
factor = htb
```
含义：当前收盘相对窗口高点与滞后收益的比较，衡量接近高位后的延续或衰减。

## fluc

Formula:
```text
N = 2^k, k in period_id_list
factor = factor
```
含义：窗口内最大最小收盘差相对价格尺度的比值，刻画震荡幅度。

## delta

Formula:
```text
N = 2^k, k in period_id_list
amt_sum_long = ts_mean(amount, (N * 2))
vol_sum_long = ts_mean(vol, (N * 2))
amt_sum_short = ts_mean(amount, N)
vol_sum_short = ts_mean(vol, N)
vwap_short = (amt_sum_short / vol_sum_short)
vwap_long = (amt_sum_long / vol_sum_long)
vwap_short = vwap_short
vwap_long = vwap_long
delta = ((vwap_short - vwap_long) / vwap_long)
factor = delta
```
含义：短长窗口 VWAP 的相对差值，反映成交重心的快慢变化。

## dens

Formula:
```text
N = 2^k, k in period_id_list
factor = factor
```
含义：路径长度相对区间跨度的密度，值越大通常代表来回震荡越充分。

## skew

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
ret_skew = ts_skew(ret, N)
factor = ret_skew
```
含义：收益率偏度，刻画分布尾部更偏向大涨还是大跌。

## std

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
ret_std = ts_std(ret, N)
factor = ret_std
```
含义：收益率标准差，衡量价格波动率。

## hl

Formula:
```text
N = 2^k, k in period_id_list
factor = (((hh / prell) + (ll / prehh)) / 2)
```
含义：当前窗口高低点与更长窗口极值的组合比值，衡量价格区间抬升或下移。

## hl2

Formula:
```text
N = 2^k, k in period_id_list
factor = (((hh / prehh) + (ll / prell)) / 2)
```
含义：当前高低点相对过去更长窗口高低点的对称位置，衡量区间整体平移。

## hl3

Formula:
```text
N = 2^k, k in period_id_list
factor = (((hh / prehh) + (ll / prell)) / 2)
```
含义：当前高低点相对更早窗口高低点的比较，强调跨周期区间变化。

## hl4

Formula:
```text
N = 2^k, k in period_id_list
factor = (((hh / prell) + (ll / prehh)) / 2)
```
含义：当前高低点相对更早窗口反向极值的比较，刻画突破或反转压力。

## ch

Formula:
```text
N = 2^k, k in period_id_list
cph = (close / high)
factor = ts_mean(cph, N)
```
含义：收盘相对高点的位置均值，越高通常越靠近区间顶部。

## cl

Formula:
```text
N = 2^k, k in period_id_list
cpl = (close / low)
factor = ts_mean(cpl, N)
```
含义：收盘相对低点的位置均值，越高通常代表离低点越远。

## lj

Formula:
```text
N = 2^k, k in period_id_list
hl = (high - low)
hlv1 = (ts_mean(hl, N) * ts_mean(vol, N))
hlv2 = ts_mean((hl * vol), N)
fz = (hlv1 + hlv2)
factor = ((hlv1 - hlv2) / fz)
```
含义：区间振幅与成交量耦合的一致性，刻画放量波动是否同步。

## upvol

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
ret_square = (ret ** 2)
up_ret_square = (0 * ret_square)
up_ret_square[ret > 0] = ret_square[ret > 0]
fenzi = ts_mean(up_ret_square, N)
fenmu = ts_mean(ret_square, N)
factor = (fenzi / fenmu)
```
含义：正收益平方在总收益平方中的占比，衡量上行波动贡献。

## upvol2

Formula:
```text
N = 2^k, k in period_id_list
ret = ((close / delay(close, 1)) - 1)
ret_square = (ret ** 2)
ret_mean = ts_mean(ret, N)
up_ret_square = (0 * ret_square)
up_ret_square[ret > ret_mean] = ret_square[ret > ret_mean]
fenzi = ts_mean(up_ret_square, N)
fenmu = ts_mean(ret_square, N)
factor = (fenzi / fenmu)
```
含义：高于均值收益部分的波动占比，衡量更强上涨样本的贡献。

## vskew

Formula:
```text
N = 2^k, k in period_id_list
factor = ts_skew(vwap, N)
```
含义：VWAP 序列偏度，刻画成交重心分布的非对称性。

## active

Formula:
```text
N = 2^k, k in period_id_list
active_sell = (vol - active_volume)
factor = ((active_buy - active_sell) / (active_buy + active_sell))
fenmu = (active_buy + active_sell)
factor[abs(np) < 1e-05] = 0
factor = ts_mean(factor, N)
```
含义：主动买量与主动卖量的相对差值，反映主动性资金偏向。

## retabs

Formula:
```text
N = 2^k, k in period_id_list
ret = (close / delay(close, 1))
ret_abs = abs(ts_mean(ret, N))
factor = ret_abs
```
含义：收益绝对值的均值，衡量平均波动幅度。

## retstd

Formula:
```text
N = 2^k, k in period_id_list
ret = (close / delay(close, 1))
ret_std = ts_std(ret, N)
factor = ret_std
```
含义：收益率标准差，衡量收益分布离散度。

## trades

Formula:
```text
N = 2^k, k in period_id_list
factor = ts_mean(trade_num, N)
```
含义：成交笔数均值，刻画交易活跃度。

## volstd

Formula:
```text
N = 2^k, k in period_id_list
vol_std = ts_std(vol, N)
vol_ma = ts_mean(vol, N)
factor = (vol_std / vol_ma)
```
含义：成交量波动率相对均量的比例，衡量量能稳定性。

## volskew

Formula:
```text
N = 2^k, k in period_id_list
volskew = ts_skew(vol, N)
factor = volskew
```
含义：成交量偏度，刻画放量尖峰是否经常出现。

## volratio

Formula:
```text
N = 2^k, k in period_id_list
vol_ma = ts_mean(vol, N)
pre_vol = delay(vol_ma, (2 * N))
factor = (vol_ma / pre_vol)
```
含义：当前均量相对更早均量的比值，衡量量能扩张或收缩。
