import numpy as np
from future_factor import FutureFactor

class MinuteMtmSustainability(FutureFactor):
    '''
    Description: auto_corr(pct_chg(close, 1), 30) * (sum(where(close > delay(close, 1), 1, 0), 30) - 15)
    Class: MTM
    Author: liuz, modified by jinpx
    '''    
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Index_Id'] = {'000300.SH':['close']}
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
                
        index_close = data['close_000300.SH'].values
        index_r = np.diff(index_close) / index_close[:-1]
        
        N = 60
        index_r_autocorr = np.corrcoef(index_r[-N:], index_r[-(N+1):-1])[0, 1]
        counter = np.sum(index_r[-N:]>0) - N/2
        f = index_r_autocorr * counter
        
        return f