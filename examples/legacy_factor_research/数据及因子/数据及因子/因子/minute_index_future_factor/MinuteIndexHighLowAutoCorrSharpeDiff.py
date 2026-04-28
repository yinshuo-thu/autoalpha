from future_factor import FutureFactor
import numpy as np


class MinuteIndexHighLowAutoCorrSharpeDiff(FutureFactor):
    data_type = 'IndexStock'
    days_past = 1
    data_dict = {}
    data_dict['Stock'] = ['close', 'adjfactor']
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        lb = 90
        adj = data['adjfactor'].values
        close = (data['close'].values * adj / adj[-1])
        rtn = close[1:] / close[:-1] - 1
        rtn = rtn[-lb:]
        nan_num = np.isnan(rtn).sum(axis=0)
        zero_num = (rtn == 0).sum(axis=0)
        rtn = rtn[:, (nan_num == 0) & (zero_num < lb / 2)]
        auto_corr = ((rtn[1:] - rtn.mean(axis=0)) * (rtn[:-1] - rtn.mean(axis=0))).mean(axis=0) / rtn.var(axis=0)
        rtn_high = rtn[:, auto_corr > np.percentile(auto_corr, 2 / 3 * 100)]
        rtn_low = rtn[:, auto_corr < np.percentile(auto_corr, 1 / 3 * 100)]
        f = rtn_high.mean() / rtn_high.std() - rtn_low.mean() / rtn_low.std()
        return f
