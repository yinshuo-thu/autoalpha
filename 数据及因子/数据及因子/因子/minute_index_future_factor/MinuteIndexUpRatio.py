import numpy as np
from future_factor import FutureFactor

class MinuteIndexUpRatio(FutureFactor):
    '''
    Description: ts_mean(cs_sum(pct_chg(close, 1) > 0), 20)
    Class: MTM
    Author: liuz, modified by jinpx
    '''
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['close', 'adjfactor']
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        close = data['close'].values
        adjfactor = data['adjfactor'].values
        close_adj = close * adjfactor

        N = 30
        r = np.diff(close_adj[-N:], axis=0) / close_adj[-N:][:-1]
        up_num = np.sum(r>0, axis=1)
        f = np.nanmean(up_num)
            
        return f