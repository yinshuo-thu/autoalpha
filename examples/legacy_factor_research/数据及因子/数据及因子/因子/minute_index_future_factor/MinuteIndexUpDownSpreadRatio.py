from future_factor import FutureFactor
import numpy as np


class MinuteIndexUpDownSpreadRatio(FutureFactor):
    '''
    Description: cs_mean(ts_mean(where(rtn < 0, spread, nan), 20) / ts_mean(where(rtn > 0, spread, nan), 20)),
                 rtn = pct_chg(close, 1),
                 spread = ask_price / bid_price,
                 ask_price = (AskP0 * AskV0 + AskP1 * AskV1) / (AskV0 + AskV1),
                 bid_price = (BidP0 * BidV0 + BidP1 * BidV1) / (BidV0 + BidV1).
    Class: Bid_Ask
    Author: hefj
    '''
    data_type = 'IndexStock'
    days_past = 1
    data_dict = {}
    data_dict['Stock'] = ['close', 'BidP0', 'AskP0', 'BidP1', 'AskP1', 'BidV0', 'AskV0', 'BidV1', 'AskV1', 'adjfactor']
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        lb = 20
        close = data['close'].values[-lb - 1:]
        close[close == 0] = np.nan
        
        ask_p = []
        bid_p = []
        ask_v = []
        bid_v = []
        for j in range(2):
            ask_p.append(data['AskP{}'.format(j)].values[-lb:])
            bid_p.append(data['BidP{}'.format(j)].values[-lb:])
            ask_v.append(data['AskV{}'.format(j)].values[-lb:])
            bid_v.append(data['BidV{}'.format(j)].values[-lb:])
        ask_p = np.array(ask_p)
        ask_p[ask_p == 0] = np.nan
        bid_p = np.array(bid_p)
        bid_p[bid_p == 0] = np.nan
        ask_v = np.array(ask_v)
        ask_v[ask_v == 0] = np.nan
        bid_v = np.array(bid_v)
        bid_v[bid_v == 0] = np.nan
        
        bid_p_mean = np.sum(bid_p * bid_v, axis=0) / np.sum(bid_v, axis=0)
        ask_p_mean = np.sum(ask_p * ask_v, axis=0) / np.sum(ask_v, axis=0)
        
        spread = ask_p_mean / bid_p_mean
        rtn = close[1:] / close[:-1] - 1
        nan_num = np.isnan(spread).sum(axis=0) + np.isnan(rtn).sum(axis=0)
        spread = spread[:, nan_num == 0]
        rtn = rtn[:, nan_num == 0]
        
        up_spread = np.nanmean(np.where(rtn > 0, spread, np.nan), axis=0)
        down_spread = np.nanmean(np.where(rtn < 0, spread, np.nan), axis=0)
        f = np.nanmean(down_spread / up_spread)

        return f
