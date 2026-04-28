from future_factor import FutureFactor
import numpy as np


class MinuteIndexHighLowWeightReturnDiff(FutureFactor):
    '''
    Description: ts_mean(cs_mean(rtn_high), 20) - ts_mean(cs_mean(rtn_low), 20),
                 rtn_high = pct_chg(close * adjfactor, 1)[:, weight[-1] > quantile(weight[-1], 0.8)],
                 rtn_low = pct_chg(close * adjfactor, 1)[:, weight[-1] < quantile(weight[-1], 0.2)].
    Class: Group_Stat
    Author: jinpx, modified by hefj
    '''
    data_type = 'IndexStock'
    days_past = 1
    data_dict = {}
    data_dict['Stock'] = ['close', 'weight', 'adjfactor']
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        weight = data['weight'].values[-1]
        close = data['close'].values[-21:]
        close[close == 0] = np.nan
        adj = data['adjfactor'].values[-21:]
        adj[adj == 0] = np.nan
        close = close * adj
        rtn = np.diff(close, axis=0) / close[:-1]
        rtn_high = np.nanmean(rtn[:, weight > np.nanquantile(weight, 0.8)], axis=1)
        rtn_low = np.nanmean(rtn[:, weight < np.nanquantile(weight, 0.2)], axis=1)
        f = np.nanmean(rtn_high) - np.nanmean(rtn_low)
        return f
