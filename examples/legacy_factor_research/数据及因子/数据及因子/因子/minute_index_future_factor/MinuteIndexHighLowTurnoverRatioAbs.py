import numpy as np
from future_factor import FutureFactor

class MinuteIndexHighLowTurnoverRatioAbs(FutureFactor):
    '''
    Description: abs(cs_mean((ts_mean(where(close > ts_quantile(close, 0.9), amount, nan), 30) 
                - ts_mean(where(close < ts_quantile(close, 0.1), amount, nan),30)) 
                / (ts_mean(where(close > ts_quantile(close, 0.9), amount, nan), 30) 
                + ts_mean(where(close < ts_quantile(close, 0.1), amount, nan), 30))))
    Class: PV_Corr
    Author: hefj, modified by lixr
    '''
    
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['weight','close', 'amount', 'adjfactor']
    normalize_size = 3 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        lb = 30
        weight = data['weight'].values
        adjfactor = data['adjfactor'].values
        close = data['close'].values * adjfactor
        close[close == 0] = np.nan
        amount = data['amount'].values
        close = close[-lb:]
        amount = amount[-lb:]
        
        high_threshold = np.nanpercentile(close, 90, axis=0)
        low_threshold = np.nanpercentile(close, 10, axis=0)
        amount_0 = np.nanmean(np.where(close > high_threshold, amount, np.nan), axis=0)
        amount_1 = np.nanmean(np.where(close < low_threshold, amount, np.nan), axis=0)
        f = np.abs(np.nanmean((amount_0 - amount_1) / (amount_0 + amount_1) * weight[-1]))
        
        if np.isnan(f) or np.isinf(f):
            return 0
        else:
            return f