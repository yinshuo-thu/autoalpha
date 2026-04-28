import numpy as np
from future_factor import FutureFactor

class MinuteIndexBuySellSuperorderCountDiff(FutureFactor):
    '''
    Description: 
    Class: 
    Author: jinpx
    '''
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['buy_superorder_count', 'sell_superorder_count']
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):

        buy_superorder_count = data['buy_superorder_count'].values
        sell_superorder_count = data['sell_superorder_count'].values
        buy_superorder_count[np.isnan(buy_superorder_count)] = 0
        sell_superorder_count[np.isnan(sell_superorder_count)] = 0
        diff = buy_superorder_count - sell_superorder_count
        
        N = 10
        f = np.nanmean(np.nanmean(diff[-N:], axis=1))
        
        return f