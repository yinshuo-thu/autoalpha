import numpy as np
import bottleneck as bn
from future_factor import FutureFactor

class MinuteIndexHighLowAbsPxPathReturnDiff(FutureFactor):
    '''
    Description: 
    Class: Group_Stat
    Author: jinpx
    '''    
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 5
    data_dict = dict()
    data_dict['Stock'] = ['close', 'adjfactor', 'AbsPxPath']
    normalize_size = 90
    normalize_type = 'ts_rank'

    def calculate(self, data):
        
        close = data['close'].values
        adjfactor = data['adjfactor'].values
        AbsPxPath = data['AbsPxPath'].values
        close_adj = close * adjfactor
        r = np.diff(close_adj, axis=0) / close_adj[:-1]
        
        N = 5 * 237
        AbsPxPath_mean = np.nanmean(AbsPxPath[-N:], axis=0)
        AbsPxPath_mean_rank = bn.rankdata(AbsPxPath_mean) / len(AbsPxPath_mean)

        r_sum = np.nansum(r[-N:], axis=0)
        
        f = np.nanmean(r_sum[AbsPxPath_mean_rank>0.8]) - np.nanmean(r_sum[AbsPxPath_mean_rank<0.2])
        
        return f