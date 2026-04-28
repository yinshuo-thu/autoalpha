import numpy as np
from future_factor import FutureFactor

class MinuteTreasureRtn120Ma(FutureFactor):
    '''
    Description: -mean(pct_chg(Treasure_LastPx, 1), 120)
    Class: Treasure_Future
    Author: shentq, modified by jinpx
    '''
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Other_Variety'] = {'T':['close']}
    normalize_size = 20 * 240
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
                
        close_T = data['close_T'].values
        r_T = np.diff(close_T) / close_T[:-1]
        
        f = - np.nanmean(r_T[-120:])
        
        return f