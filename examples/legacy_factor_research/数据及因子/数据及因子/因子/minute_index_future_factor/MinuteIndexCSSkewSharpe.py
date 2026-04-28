import numpy as np
from future_factor import FutureFactor

class MinuteIndexCSSkewSharpe(FutureFactor):
    '''
    Description: weighted_ts_mean(weighted_cs_skew(pct_chg(close, 1), w=index_weight), 30, w=range(1, 31))
                / weighted_ts_std(weighted_cs_skew(pct_chg(close, 1), w=index_weight), 30, w=range(1, 31))
    Class: Price_CS_Stat
    Author: hefj, modified by lixr
    '''
    
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['weight', 'close', 'adjfactor']
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        lb = 30
        w = np.arange(1, lb + 1)
        w = w / w.sum()
        
        weight = data['weight'].values
        adjfactor = data['adjfactor'].values
        close = data['close'].values * adjfactor
        close[close == 0] = np.nan
        
        rtn_temp = close[-lb:] / close[-lb - 1: -1] - 1
        mean = np.nansum(rtn_temp * weight[-lb:], axis=1).reshape((len(rtn_temp), -1))
        std = np.nansum(((rtn_temp - mean) ** 2) * weight[-lb:], axis=1) ** 0.5
        skew = np.nansum(((rtn_temp - mean) ** 3) * weight[-lb:], axis=1) / (std ** 3)
        mean = np.sum(skew * w)
        std = np.sum(((skew - mean) ** 2) * w) ** 0.5
        
        return mean / std