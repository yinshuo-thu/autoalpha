import numpy as np
from future_factor import FutureFactor
import bottleneck as bn

class MinuteIndexHighLowTurnoverRateReturnDiff(FutureFactor):
    '''
    Description: 
    Class: 
    Author: jinpx
    '''    
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 5
    data_dict = dict()
    data_dict['Stock'] = ['turnover_rate', 'close', 'adjfactor']
    normalize_size = 60
    normalize_type = 'ts_rank'

    def calculate(self, data):
        
        turnover_rate = data['turnover_rate'].values
        close = data['close'].values
        adjfactor = data['adjfactor'].values
        close_adj = close * adjfactor
        r = np.diff(close_adj, axis=0) / close_adj[:-1]
        
        N = 5 * 237
        turnover_rate_mean = np.nanmean(turnover_rate[-N:], axis=0)
        turnover_rate_mean_rank = bn.rankdata(turnover_rate_mean) / len(turnover_rate_mean)
        
        r_sum = np.nansum(r[-N:], axis=0)
        
        f = np.nanmean(r_sum[turnover_rate_mean_rank>0.8]) - np.nanmean(r_sum[turnover_rate_mean_rank<0.2])
        
        return f