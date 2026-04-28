import numpy as np
from future_factor import FutureFactor

class MinuteVR1Corr15Sr120(FutureFactor):
    '''
    Description: mean(corr(pct_chg(close, 1), volume, 15), 120) / std(corr(pct_chg(close, 1), volume, 15), 120)
    Class: PV_Corr
    Author: liuz, modified by jinpx
    '''    
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Continuous_Data'] = {'IF':['close', 'volume']}
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def rolling_corr(self, data_1, data_2, window):
        assert len(data_1)==len(data_2), 'length of two arrays must be the same! ({}, {})'.format(len(data_1), len(data_2))
        rolling_corr = np.array([])
        for i in range(len(data_1) - window + 1):
            rolling_corr = np.append(rolling_corr, np.corrcoef(data_1[i:i+window], data_2[i:i+window])[0, 1])
        return rolling_corr
    
    def calculate(self, data):
              
        volume = data['volume_cont_IF'].values  
        close = data['close_cont_IF'].values
        r = (close[1:] - close[:-1]) / close[:-1]
        r_volume_corr_15 = self.rolling_corr(r, volume[1:], 15)
        N = 120
        f = np.nanmean(r_volume_corr_15[-N:]) / np.nanstd(r_volume_corr_15[- N:], ddof=1)
        
        return f