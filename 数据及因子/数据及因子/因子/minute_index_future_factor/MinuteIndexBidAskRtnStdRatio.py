from future_factor import FutureFactor
import numpy as np


class MinuteIndexBidAskRtnStdRatio(FutureFactor):
    '''
    Description: cs_weighted_mean((bid_std - ask_std) / (bid_std + ask_std), w=weight),
                 bid_std = ts_std(pct_chg(bid_price, 1), 10),
                 ask_std = ts_std(pct_chg(ask_price, 1), 10),
                 bid_price = (BidP0 * BidV0 + BidP1 * BidV1 + BidP2 * BidV2 + BidP3 * BidV3 + BidP4 * BidV4) / (BidV0 + BidV1 + BidV2 + BidV3 + BidV4) * adjfactor,
                 ask_price = (AskP0 * AskV0 + AskP1 * AskV1 + AskP2 * AskV2 + AskP3 * AskV3 + AskP4 * AskV4) / (AskV0 + AskV1 + AskV2 + AskV3 + AskV4) * adjfactor.
    Class: Bid_Ask
    Author: hefj
    '''
    data_type = 'IndexStock'
    days_past = 1
    data_dict = {}
    data_dict['Stock'] = ['BidV0', 'BidV1', 'BidV2', 'BidV3', 'BidV4', 'AskV0', 'AskV1', 'AskV2', 'AskV3', 'AskV4',
                         'BidP0', 'BidP1', 'BidP2', 'BidP3', 'BidP4', 'AskP0', 'AskP1', 'AskP2', 'AskP3', 'AskP4', 'adjfactor', 'weight']
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        lb = 11
        
        adj = data['adjfactor'].values[-lb:]
        weight = data['weight'].values[-1]
        
        bid_p = []
        bid_v = []
        ask_p = []
        ask_v = []
        
        for j in range(5):
            bid_p.append(data['BidP{}'.format(j)].values[-lb:])
            bid_v.append(data['BidV{}'.format(j)].values[-lb:])
            ask_p.append(data['AskP{}'.format(j)].values[-lb:])
            ask_v.append(data['AskV{}'.format(j)].values[-lb:])
            
        bid_p = np.array(bid_p)
        bid_p[bid_p == 0] = np.nan
        bid_v = np.array(bid_v)
        bid_v[bid_v == 0] = np.nan
        ask_p = np.array(ask_p)
        ask_p[ask_p == 0] = np.nan
        ask_v = np.array(ask_v)
        ask_v[ask_v == 0] = np.nan
        
        bid_p_mean = np.nansum(bid_p * bid_v, axis=0) / np.nansum(bid_v, axis=0) * adj
        bid_rtn = bid_p_mean[1:] / bid_p_mean[:-1] - 1
        
        ask_p_mean = np.nansum(ask_p * ask_v, axis=0) / np.nansum(ask_v, axis=0) * adj
        ask_rtn = ask_p_mean[1:] / ask_p_mean[:-1] - 1
        
        inf_nan_num = np.isnan(bid_rtn).sum(axis=0) + np.isinf(bid_rtn).sum(axis=0) + np.isnan(ask_rtn).sum(axis=0) + np.isinf(ask_rtn).sum(axis=0)
        bid_std = np.std(bid_rtn[:, inf_nan_num == 0], axis=0)
        ask_std = np.std(ask_rtn[:, inf_nan_num == 0], axis=0)
        f = np.nansum((bid_std - ask_std) / (bid_std + ask_std) * weight[inf_nan_num == 0])
        return f
