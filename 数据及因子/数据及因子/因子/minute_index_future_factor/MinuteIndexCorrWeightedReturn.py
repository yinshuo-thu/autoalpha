import numpy as np
from future_factor import FutureFactor

class MinuteIndexCorrWeightedReturn(FutureFactor):
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
    normalize_size = 5 * 237
    normalize_type = 'ts_rank'

    def calculate(self, data):
        
        stk_index_corr_hs300 = data['stk_index_corr_hs300'].values
        close = data['close'].values
        adjfactor = data['adjfactor'].values
        close_adj = close * adjfactor
        r = np.diff(close_adj, axis=0) / close_adj[-1]
        
        f = np.nanmean(np.nansum(r[-1185:], axis=0) * stk_index_corr_hs300[-1])
        
        return f