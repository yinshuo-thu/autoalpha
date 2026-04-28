import numpy as np
from future_factor import FutureFactor

class MinuteIndexHighVolumeCorr(FutureFactor):
    '''
    Description: abs(cs_mean(ts_corr(high, volume, 60)))
    Class: PV_Corr
    Author: jinpx, modified by jinpx
    '''
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['high', 'volume', 'adjfactor']
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        high = data['high'].values
        volume = data['volume'].values
        adjfactor = data['adjfactor'].values
        high_adj = high * adjfactor
        volume_adj = volume / adjfactor
        
        N = 60
        c = np.array([])
        for i in range(len(high_adj[-1])):
            c = np.append(c, np.corrcoef(high_adj[-N:,i], volume_adj[-N:,i])[0,1])
        
        f = np.abs(np.nanmean(c))
        
        return f