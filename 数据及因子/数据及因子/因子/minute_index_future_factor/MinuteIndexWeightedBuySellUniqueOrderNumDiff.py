import numpy as np
from future_factor import FutureFactor

class MinuteIndexWeightedBuySellUniqueOrderNumDiff(FutureFactor):
    '''
    Description: 
    Class: 
    Author: jinpx
    '''    
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['BuyUniqueOrderNum', 'SellUniqueOrderNum', 'weight']
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'

    def calculate(self, data):

        BuyUniqueOrderNum = data['BuyUniqueOrderNum'].values
        SellUniqueOrderNum = data['SellUniqueOrderNum'].values
        weight = data['weight'].values
        
        BuySellUniqueDiff = BuyUniqueOrderNum - SellUniqueOrderNum

        N = 120
        BuySellUniqueDiff_mean = np.nanmean(BuySellUniqueDiff[-N:], axis=0)
        f = - np.nansum(weight[-1] * BuySellUniqueDiff_mean)
        
        return f