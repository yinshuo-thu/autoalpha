import numpy as np
from future_factor import FutureFactor

class MinuteIndexNewHighNewLowSizeDiff(FutureFactor):
    '''
    Description: 
    Class: 
    Author: jinpx
    '''    
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 3
    data_dict = dict()
    data_dict['Stock'] = ['close', 'high', 'low', 'adjfactor', 'float_shares']
    normalize_size = 237
    normalize_type = 'ts_rank'

    def calculate(self, data):
        
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        adjfactor = data['adjfactor'].values
        float_shares = data['float_shares'].values
        close_adj = close * adjfactor
        high_adj = high * adjfactor
        low_adj = low * adjfactor
        
        high_num = np.nansum((np.nanmax(high_adj[-60:], axis=0) > np.max(high_adj[-3*237:-60], axis=0)) * close_adj[-1] * float_shares[-1])
        low_num = np.nansum((np.nanmax(low_adj[-60:], axis=0) < np.min(low_adj[-3*237:-60], axis=0)) * close_adj[-1] * float_shares[-1])
        
        f = high_num - low_num
        
        return f