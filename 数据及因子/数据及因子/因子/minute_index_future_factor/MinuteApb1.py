import numpy as np
from future_factor import FutureFactor

class MinuteApb1(FutureFactor):
    '''
    Description: mean(amount / volume, 95) / (sum(amount, 95) / sum(volume, 95))
    Class: PV_Corr
    Author: liuz, modified by jinpx
    '''
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Continuous_Data'] = {'IF':['volume', 'amount']}
    normalize_size = 35 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
                
        volume = data['volume_cont_IF'].values
        amount = data['amount_cont_IF'].values
        vwap = amount / volume
        N = 95
        f = np.nanmean(vwap[-N:]) / (np.nansum(amount[-N:]) / np.nansum(volume[-N:]))
        
        return f