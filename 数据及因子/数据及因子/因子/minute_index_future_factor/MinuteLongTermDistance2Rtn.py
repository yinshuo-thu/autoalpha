import numpy as np
from future_factor import FutureFactor

class MinuteLongTermDistance2Rtn(FutureFactor):
    '''
    Description: Sum(r, 240) / Sum(Abs(r), 240)
    Class: Return_Risk
    Author: liuz, modified by jinpx
    '''    
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Index_Id'] = {'000300.SH':['close', 'high', 'low']}
    normalize_size = 60
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
                
        index_close = data['close_000300.SH'].values
        index_high = data['high_000300.SH'].values
        index_low = data['low_000300.SH'].values
        index_typical = index_close + index_high + index_low
        index_typical_r = np.diff(index_typical) / index_typical[:-1]
        
        N = 240
        f = np.sum(index_typical_r[-N:]) / np.sum(np.abs(index_typical_r[-N:]))

        return f