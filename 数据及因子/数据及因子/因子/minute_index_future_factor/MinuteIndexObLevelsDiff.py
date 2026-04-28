from future_factor import FutureFactor
import numpy as np


class  MinuteIndexObLevelsDiff(FutureFactor):
    '''
    Description: 
    Class: Bid_Ask
    Author: shentq  modeified by liuz
    '''
    data_type = 'IndexStock'
    instrument_type='main'
    days_past=1
    data_dict=dict()
    data_dict['Stock'] = ['TotalBidVol', 'TotalAskVol', 'BidVolMean', 'AskVolMean']

    normalize_size=1*237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):

        TotalBidVol = data['TotalBidVol'].values
        TotalAskVol = data['TotalAskVol'].values
        BidVolMean = data['BidVolMean'].values
        AskVolMean = data['AskVolMean'].values

        bid_levels = TotalBidVol / BidVolMean
        ask_levels = TotalAskVol / AskVolMean
        
    
        bid_levels[np.isnan(bid_levels) | np.isinf(bid_levels)] = np.nan
        ask_levels[np.isnan(ask_levels) | np.isinf(ask_levels)] = np.nan

        factor = np.nanmean(np.nanmean((bid_levels-ask_levels)[-120:],axis=0))
        
        return factor
