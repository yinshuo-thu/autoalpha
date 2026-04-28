import numpy as np
from future_factor import FutureFactor

class MinuteIndexBidAskSpreadMeanRatioSum(FutureFactor):
    '''
    Description: BidAskSpreadMean / (Mean of BidAskSpreadMean during past 5 days at the same minute)
    Class: 
    Author: jinpx
    '''    
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 6
    data_dict = dict()
    data_dict['Stock'] = ['BidAskSpreadMean']
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'

    def calculate(self, data):
        
        BidAskSpreadMean = data['BidAskSpreadMean'].values
        
        BidAskSpreadMean_ratio = BidAskSpreadMean[-237:] / np.nanmean(BidAskSpreadMean[-1422:-237].reshape(5, 237, len(BidAskSpreadMean[0])), axis=0)
        
        N = 3
        f = - np.nansum(BidAskSpreadMean_ratio[-N:])
        
        return f