import numpy as np
from future_factor import FutureFactor

class MinuteConsecutiveUpRatio40(FutureFactor):
    '''
    Description: sum(where((delay(close_000300.SH, 2) < delay(close_000300.SH, 1)) & (close_000300.SH < delay(close_000300.SH, 1)), 1, 0), 40)
                / sum(where(delay(close_000300.SH, 1) < close_000300.SH, 1, 0), 40)
    Class: MTM
    Author: hefj, modified by lixr
    '''
    
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Index_Id'] = {'000300.SH':['close']}
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        lb = 40
        close = data['close_000300.SH'].values
        close[close == 0] = np.nan
        close_temp = close[-lb:]
        close_temp = close_temp[~np.isnan(close_temp)]
        
        up = (close_temp[1:] > close_temp[:-1]).sum()
        consecutive_up = ((close_temp[:-2] < close_temp[1: -1]) & (close_temp[2:] > close_temp[1: -1])).sum()
       
        return consecutive_up / up