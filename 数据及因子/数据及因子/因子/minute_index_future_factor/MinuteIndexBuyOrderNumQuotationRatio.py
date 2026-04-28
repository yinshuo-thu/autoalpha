import numpy as np
from future_factor import FutureFactor

class MinuteIndexBuyOrderNumQuotationRatio(FutureFactor):
    '''
    Description: 
    Class: 
    Author: jinpx 
    '''    
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['Buy1NumOrdersMean', 'BuyNumOrdersSumMean']
    normalize_size = 120
    normalize_type = 'ts_rank'

    def calculate(self, data):
        
        Buy1NumOrdersMean = data['Buy1NumOrdersMean'].values
        BuyNumOrdersSumMean = data['BuyNumOrdersSumMean'].values
        
        buy_ratio = BuyNumOrdersSumMean / Buy1NumOrdersMean
        
        N = 5        
        f = - np.nanmean(buy_ratio[-N:])
        
        return f