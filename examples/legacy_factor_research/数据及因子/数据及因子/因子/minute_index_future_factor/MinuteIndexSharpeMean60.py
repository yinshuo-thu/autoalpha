from future_factor import FutureFactor
import numpy as np


class MinuteIndexSharpeMean60(FutureFactor):
    '''
    Description: cs_weighted_mean(ts_mean(rtn, 60) / ts_std(rtn, 60), w=weight),
                 rtn = pct_chg(close * adjfactor, 1).
    Class: MTM
    Author: hefj
    '''
    data_type = 'IndexStock'
    days_past = 1
    data_dict = {}
    data_dict['Stock'] = ['close', 'adjfactor', 'weight']
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        lb = 60
        weight = data['weight'].values[-1]
        close = data['close'].values[-lb - 1:]
        close[close == 0] = np.nan
        adj = data['adjfactor'].values[-lb - 1:]
        adj[adj == 0] = np.nan
        close = close * adj
        rtn = close[1:] / close[:-1] - 1
        inf_nan_num = np.isnan(rtn).sum(axis=0) + np.isinf(rtn).sum(axis=0)
        rtn = rtn[:, inf_nan_num == 0]
        sharpe = np.nanmean(rtn, axis=0) / np.nanstd(rtn, axis=0)
        f = np.nansum(sharpe * weight[inf_nan_num == 0])
        return f
