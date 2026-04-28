from future_factor import FutureFactor
import numpy as np


class MinuteIndexOrderImbalanceStd(FutureFactor):
    '''
    Description: -ts_mean(cs_std((bid - ask) / (bid + ask)), 5),
                 bid = BidV0 + BidV1 + BidV2 + BidV3 + BidV4,
                 ask = AskV0 + AskV1 + AskV2 + AskV3 + AskV4.
    Class: Bid_Ask
    Author: jinpx, modified by hefj
    '''
    data_type = 'IndexStock'
    days_past = 1
    data_dict = {}
    data_dict['Stock'] = ['BidV0', 'BidV1', 'BidV2', 'BidV3', 'BidV4', 'AskV0', 'AskV1', 'AskV2', 'AskV3', 'AskV4']
    normalize_size = 10 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        lb = 5
        bid0 = data['BidV0'].values[-lb:]
        bid0[bid0 == 0] = np.nan
        bid1 = data['BidV1'].values[-lb:]
        bid1[bid1 == 0] = np.nan
        bid2 = data['BidV2'].values[-lb:]
        bid2[bid2 == 0] = np.nan
        bid3 = data['BidV3'].values[-lb:]
        bid3[bid3 == 0] = np.nan
        bid4 = data['BidV4'].values[-lb:]
        bid4[bid4 == 0] = np.nan
        ask0 = data['AskV0'].values[-lb:]
        ask0[ask0 == 0] = np.nan
        ask1 = data['AskV1'].values[-lb:]
        ask1[ask1 == 0] = np.nan
        ask2 = data['AskV2'].values[-lb:]
        ask2[ask2 == 0] = np.nan
        ask3 = data['AskV3'].values[-lb:]
        ask3[ask3 == 0] = np.nan
        ask4 = data['AskV4'].values[-lb:]
        ask4[ask4 == 0] = np.nan
        bid = bid0 + bid1 + bid2 + bid3 + bid4
        ask = ask0 + ask1 + ask2 + ask3 + ask4
        order_imbalance = (bid - ask) / (bid + ask)
        f = -np.nanmean(np.nanstd(order_imbalance, axis=1))
        return f
