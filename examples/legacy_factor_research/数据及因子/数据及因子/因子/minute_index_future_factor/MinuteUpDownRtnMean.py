import numpy as np
from future_factor import FutureFactor

class MinuteUpDownRtnMean(FutureFactor):
    '''
    Description: (close_000300.SH / min(close_000300.SH, 60) - 1) / (60 - argmin(close_000300.SH, 60)) 
                + (close_000300.SH / max(close_000300.SH, 60) - 1) / (60 - argmax(close_000300.SH, 60))
    Class: MTM
    Author: hefj, modified by lixr
    '''
    
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Index_Id'] = {'000300.SH':['close']}
    normalize_size = 10 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        lb = 60
        close = data['close_000300.SH'].values
        close[close == 0] = np.nan
        close_temp = close[-lb:]
        
        argmin = np.nanargmin(close_temp)
        argmax = np.nanargmax(close_temp)
        f = (close_temp[-1] / np.nanmin(close_temp) - 1) / (lb - argmin) + (close_temp[-1] / np.nanmax(close_temp) - 1) / (lb - argmax)
        
        if np.isnan(f) or np.isinf(f):
            return 0
        else:
            return f