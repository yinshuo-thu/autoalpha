import numpy as np
from future_factor import FutureFactor
from scipy.stats import skew

class MinuteIndexTSSkewSharpe(FutureFactor):
    '''
    Description: cs_mean(ts_skew(pct_chg(close, 1), 30)) / cs_std(ts_skew(pct_chg(close, 1), 30))
    Class: Price_TS_Stat
    Author: hefj, modified by lixr
    '''
    
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['adjfactor', 'close']
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        lb = 30
        adjfactor = data['adjfactor'].values
        close = data['close'].values * adjfactor
        close[close == 0] = np.nan
        
        rtn = close[1:] / close[:-1] - 1
        rtn_temp = rtn[-lb:]
        skew_temp = skew(rtn_temp, axis=0, nan_policy='omit')
        
        return np.nanmean(skew_temp) / np.nanstd(skew_temp)