import numpy as np
import bottleneck as bn
from future_factor import FutureFactor

class MinuteIndexHighLowCorrReturnDiff(FutureFactor):
    '''
    Description: 
    Class: 
    Author: jinpx
    '''    
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 5
    data_dict = dict()
    data_dict['Stock'] = ['stk_index_corr_hs300', 'close', 'adjfactor']
    normalize_size = 60
    normalize_type = 'ts_rank'

    def calculate(self, data):
        
        stk_index_corr_hs300 = data['stk_index_corr_hs300'].values
        close = data['close'].values
        adjfactor = data['adjfactor'].values
        close_adj = close * adjfactor
        r = np.diff(close_adj, axis=0) / close_adj[:-1]
        
        N = 5 * 237
        stk_index_corr_hs300_mean = np.nanmean(stk_index_corr_hs300[-N:], axis=0)
        stk_index_corr_hs300_mean_rank = bn.rankdata(stk_index_corr_hs300_mean) / len(stk_index_corr_hs300_mean)
        
        r_sum = np.nansum(r[-N:], axis=0)
        
        f = np.nanmean(r_sum[stk_index_corr_hs300_mean_rank>0.8]) - np.nanmean(r_sum[stk_index_corr_hs300_mean_rank<0.2])
        
        return f