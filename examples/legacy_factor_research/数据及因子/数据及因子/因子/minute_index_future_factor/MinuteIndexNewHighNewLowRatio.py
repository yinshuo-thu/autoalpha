import numpy as np
from future_factor import FutureFactor

class MinuteIndexNewHighNewLowRatio(FutureFactor):
    '''
    Description:(cs_sum(ts_max(close, 240) == close[-1]) - cs_sum(ts_min(close, 240) == close[-1]))
                / (cs_sum(ts_max(close, 240) == close[-1]) + cs_sum(ts_min(close, 240) == close[-1]))    
    Class: MTM
    Author: hefj, modified by lixr
    '''
    
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['adjfactor', 'close']
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        lb = 237
        adjfactor = data['adjfactor'].values
        close = data['close'].values * adjfactor
        close[close == 0] = np.nan
        
        new_high = np.where(np.nanmax(close[-lb:], axis=0) == close[-1], 1, 0).sum()
        new_low = np.where(np.nanmin(close[-lb:], axis=0) == close[-1], 1, 0).sum()
        
        if (new_high + new_low) == 0:
            return 0
        else:
            return (new_high - new_low) / (new_high + new_low)