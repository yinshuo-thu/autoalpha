import numpy as np
from future_factor import FutureFactor

class MinuteUpDownAmountDiff(FutureFactor):
    '''
    Description: sum(where(index_close >= delay(index_close, 1), index_amount, 0), 90) / sum(index_amount, 90)
    Class: MTM
    Author: liuz, modified by jinpx
    '''   
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Index_Id'] = {'000300.SH':['close', 'amount']}
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        N = 90
        index_amount = data['amount_000300.SH'].values[-N:]
        index_close = data['close_000300.SH'].values[-N:]
        
        index_r = np.append(np.nan, np.diff(index_close) / index_close[:-1])
        f = np.nansum(index_amount[index_r>=0]) / np.nansum(index_amount)        
        
        return f