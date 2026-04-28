from future_factor import FutureFactor
import numpy as np


class MinuteIndexAskBidDepthRatio(FutureFactor):
    '''
    Description: ts_mean(cs_mean((BidP0 / BidP4 - 1) / (AskP4 / AskP0 - 1)), 10)
    Class: Bid_Ask
    Author: hefj
    '''
    data_type = 'IndexStock'
    days_past = 1
    data_dict = {}
    data_dict['Stock'] = ['BidP0', 'BidP4', 'AskP0', 'AskP4']
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        bid_0 = data['BidP0'].values[-10:]
        bid_0[bid_0 == 0] = np.nan
        bid_4 = data['BidP4'].values[-10:]
        bid_4[bid_4 == 0] = np.nan
        ask_0 = data['AskP0'].values[-10:]
        ask_0[ask_0 == 0] = np.nan
        ask_4 = data['AskP4'].values[-10:]
        ask_4[ask_4 == 0] = np.nan
        ratio = (bid_0 / bid_4 - 1) / (ask_4 / ask_0 - 1)
        ratio[np.isinf(ratio)] = np.nan
        f = np.nanmean(ratio)
        return f
