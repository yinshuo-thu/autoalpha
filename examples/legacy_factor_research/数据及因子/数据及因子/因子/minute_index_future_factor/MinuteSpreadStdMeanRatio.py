import numpy as np
from future_factor import FutureFactor

class MinuteSpreadStdMeanRatio(FutureFactor):
    '''
    Description: BidAskVol / BidAskMean
    Class: 
    Author: jinpx
    '''    
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Future_Data'] = ['BidAskVol', 'BidAskMean']
    normalize_size = 120
    normalize_type = 'ts_rank'

    def calculate(self, data):

        BidAskVol = data['BidAskVol'].values
        BidAskMean = data['BidAskMean'].values
        
        SpreadStdMeanRatio = BidAskVol / BidAskMean
        
        N = 20
        f = np.nanmean(SpreadStdMeanRatio[-N:])
        
        return f