import numpy as np
from future_factor import FutureFactor

class MinuteIndexRtnAutoCorr(FutureFactor):
    '''
    Description: Sum(AutoCorr(r, 30) * weight)
    Class: Price_TS_Stat
    Author: hefj, modified by jinpx
    '''
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['close', 'adjfactor', 'weight']
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        close = data['close'].values
        adjfactor = data['adjfactor'].values
        weight = data['weight'].values
        
        close_adj = close * adjfactor   
        r = np.diff(close_adj, axis=0) / close_adj[:-1] 
        
        N = 30
        r = r[-N:]
        auto_corr = np.nanmean((r[1:] - np.nanmean(r[1:], axis=0)) * (r[:-1] - np.nanmean(r[:-1], axis=0)), axis=0) / np.nanstd(r[1:], axis=0) / np.nanstd(r[:-1], axis=0)
        f = np.nansum(auto_corr * weight[-1])
        if np.isnan(f) or np.isinf(f):
            f = 0
            
        return f