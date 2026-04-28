import numpy as np
from future_factor import FutureFactor

class MinuteIndexUpDownRtnMean(FutureFactor):
    '''
    Description: weighted_cs_mean(((close[-1] / ts_max(close, 240) - 1) / (240 - ts_arg_max(close, 240)) 
                + (close[-1] / ts_min(close, 240) - 1) / (240 - ts_argmin(close, 240))) / 2, w=index_weight)
    Class: MTM
    Author: hefj, modified by lixr
    '''
    
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['weight', 'adjfactor', 'close']
    normalize_size = 10 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        lb = 237
        weight = data['weight'].values
        adjfactor = data['adjfactor'].values
        close = data['close'].values * adjfactor
        close[close == 0] = np.nan
        
        close_price = close[-lb:]
        mask = np.all(np.isnan(close_price), axis=0)
        close_price = close_price[:, ~mask]
        argmax = lb - np.nanargmax(close_price, axis=0)
        rtn_0 = (close_price[-1] / np.nanmax(close_price, axis=0) - 1) / argmax
        argmin = lb - np.nanargmin(close_price, axis=0)
        rtn_1 = (close_price[-1] / np.nanmin(close_price, axis=0) - 1) / argmin
        rtn = (rtn_1 + rtn_0) / 2
        f = np.nansum(rtn * weight[-1][~mask])
        
        if np.isnan(f) or np.isinf(f):
            return 0
        else:
            return f