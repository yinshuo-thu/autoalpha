import numpy as np
from future_factor import FutureFactor

class MinuteOpenCloseCorr(FutureFactor):
    '''
    Description: corr(open, close, 60)
    Class: Price_Stat
    Author: jinpx, modified by jinpx
    '''    
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Index_Id'] = {'000300.SH':['open', 'close']}
    normalize_size = 5 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
                
        index_open = data['open_000300.SH'].values
        index_close = data['close_000300.SH'].values
        
        N = 45
        f = np.corrcoef(index_open[-N:], index_close[-N:])[0, 1]
        
        return f