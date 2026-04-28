import numpy as np
import bottleneck as bn
from future_factor import FutureFactor

class MinuteIFIHReturnDiff(FutureFactor):
    '''
    Description: 
    Class: 
    Author: 
    '''    
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Index_Id'] = {'000300.SH':['close'], '000016.SH':['close']}
    normalize_size = 40 * 237
    normalize_type = 'ts_rank'

    def calculate(self, data):
        
        close_if = data['close_000300.SH'].values
        close_ih = data['close_000016.SH'].values
        
        r_if = np.diff(close_if) / close_if[:-1]
        r_ih = np.diff(close_ih) / close_ih[:-1]

        N = 20
        f = np.nanmean(r_if[-N:]) - np.nanmean(r_ih[-N:])
        
        return f