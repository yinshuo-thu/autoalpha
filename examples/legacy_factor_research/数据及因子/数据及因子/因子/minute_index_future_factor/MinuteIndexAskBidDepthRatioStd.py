from future_factor import FutureFactor
import numpy as np


class MinuteIndexAskBidDepthRatioStd(FutureFactor):
    '''
    Description: -ts_mean(cs_std((AskP4 - AskP0) / (BidP0 - BidP4)), 30)
    Class: Bid_Ask
    Author: jinpx, modified by hefj
    '''
    data_type = 'IndexStock'
    days_past = 1
    data_dict = {}
    data_dict['Stock'] = ['BidP0', 'BidP4', 'AskP0', 'AskP4']
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        bid_0 = data['BidP0'].values[-30:]
        bid_0[bid_0 == 0] = np.nan
        bid_4 = data['BidP4'].values[-30:]
        bid_4[bid_4 == 0] = np.nan
        ask_0 = data['AskP0'].values[-30:]
        ask_0[ask_0 == 0] = np.nan
        ask_4 = data['AskP4'].values[-30:]
        ask_4[ask_4 == 0] = np.nan
        ratio = (ask_4 - ask_0) / (bid_0 - bid_4)
        ratio[np.isinf(ratio)] = np.nan
        f = -np.nanmean(np.nanstd(ratio, axis=1))
        return f
