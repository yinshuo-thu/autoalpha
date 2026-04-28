from future_factor import FutureFactor
import numpy as np


class MinuteIndexHighLowStdRtnDiff(FutureFactor):
    data_type = 'IndexStock'
    days_past = 1
    data_dict = {}
    data_dict['Stock'] = ['close', 'adjfactor']
    normalize_size = 120
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        lb = 237
        adj = data['adjfactor'].values
        close = (data['close'].values * adj / adj[-1])[-lb - 1:]
        rtn = close[1:] / close[:-1] - 1
        zero_num = (np.abs(rtn) < 1 / 10000).sum(axis=0)
        rtn = rtn[:, zero_num < (lb / 2)]
        std = np.std(rtn, axis=0)
        median = np.nanmedian(std)
        rtn_high_std = rtn[:, std > median]
        rtn_low_std = rtn[:, std < median]
        f = np.nanmean(rtn_high_std) - np.nanmean(rtn_low_std)
        return f
