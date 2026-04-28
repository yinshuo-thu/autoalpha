import numpy as np
from future_factor import FutureFactor

class MinuteDownsideReSR30(FutureFactor):
    '''
    Description: mean(where(pct_chg(index_close, 1) > 0, 0, pct_chg(index_close, 1)), 30) / std(where(pct_chg(index_close, 1) > 0, 0, pct_chg(index_close, 1)), 30)
    Class: Return_Risk
    Author: liuz, modified by jinpx
    '''  
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Index_Id'] = {'000300.SH':['close']}
    normalize_size = 5 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        index_close = data['close_000300.SH'].values
        index_r = np.diff(index_close) / index_close[:-1]
        index_r[index_r>=0] = 0
        N = 30
        f = np.nanmean(index_r[-N:]) / np.nanstd(index_r[-N:])
        if np.isnan(f):
            f = 0
        return f