from future_factor import FutureFactor
import numpy as np


class MinuteLongTermRtn(FutureFactor):
    '''
    Description: mean(rtn, from 09:30 T-5),
                 rtn = pct_chg(close_000300.SH, 1)
    Class: MTM
    Author: hefj
    '''
    data_type = 'Future'
    days_past = 5
    data_dict = {}
    data_dict['Index_Id'] = {'000300.SH': ['close']}
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        close = data['close_000300.SH'].values
        rtn = close[1:] / close[:-1] - 1
        f = np.nanmean(rtn)
        return f
