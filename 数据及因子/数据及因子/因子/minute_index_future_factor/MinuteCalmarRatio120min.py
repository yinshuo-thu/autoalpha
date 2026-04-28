import numpy as np
from future_factor import FutureFactor

class MinuteCalmarRatio120min(FutureFactor):
    '''
    Description: calmar(index_close, 120)
    Class: Return_Risk
    Author: liuz, modified by jinpx
    '''    
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Index_Id'] = {'000300.SH':['close']}
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
                
        index_close = data['close_000300.SH'].values
        
        N = 120
        index_r = np.diff(index_close[-N:]) / index_close[-N:][:-1]
        index_r_cumsum = np.cumsum(index_r) + 1
        
        max_r = 0
        max_drawdown = 0.0000001
        
        for i in range(N-1):
            if index_r_cumsum[i] > max_r:
                max_r = index_r_cumsum[i]
            if index_r_cumsum[i] / max_r - 1 < max_drawdown:
                max_drawdown = index_r_cumsum[i] / max_r - 1 
        f = - index_r.mean() / max_drawdown
        
        return f