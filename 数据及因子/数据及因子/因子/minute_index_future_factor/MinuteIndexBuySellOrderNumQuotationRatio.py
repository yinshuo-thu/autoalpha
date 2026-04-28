import numpy as np
from future_factor import FutureFactor

class MinuteIndexBuySellOrderNumQuotationRatio(FutureFactor):
    '''
    Description: 
    Class: 
    Author: jinpx 
    '''    
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['Buy1NumOrdersMean', 'Sell1NumOrdersMean', 'BuyNumOrdersSumMean', 'SellNumOrdersSumMean']
    normalize_size = 237
    normalize_type = 'ts_rank'

    def calculate(self, data):
        
        Buy1NumOrdersMean = data['Buy1NumOrdersMean'].values
        Sell1NumOrdersMean = data['Sell1NumOrdersMean'].values
        BuyNumOrdersSumMean = data['BuyNumOrdersSumMean'].values
        SellNumOrdersSumMean = data['SellNumOrdersSumMean'].values
        
        buy_ratio = BuyNumOrdersSumMean / Buy1NumOrdersMean
        sell_ratio = SellNumOrdersSumMean / Sell1NumOrdersMean
        
        N = 20
        f = - (np.nanmean(buy_ratio[-N:]) - np.nanmean(sell_ratio[-N:]))
        
        return f