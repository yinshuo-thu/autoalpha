import numpy as np
from future_factor import FutureFactor
from scipy.stats import skew

class MinuteIndexNetNewBidAskDiffSkew(FutureFactor):
    '''
    Description: sum(skew(new_bid - new_ask, 240), w = index_weight), 
                where new_bid = (diff(bid) + sell) / adjfactor, new_ask = (diff(ask) + buy) / adjfactor
    Class: Bid_Ask
    Author: hefj, modified by lixr
    '''
    
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['weight', 'adjfactor', 'BuyTradeQuantity', 'SellTradeQuantity', 'TotalBidVol', 'TotalAskVol']
    normalize_size = 3 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        lb = 237
        weight = data['weight'].values
        adjfactor = data['adjfactor'].values
        buy = data['BuyTradeQuantity'].values
        sell = data['SellTradeQuantity'].values
        bid = data['TotalBidVol'].values
        ask = data['TotalAskVol'].values
        new_bid = (np.diff(bid[:237], axis=0) + sell[1:237]) / adjfactor[1:237]
        new_ask = (np.diff(ask[:237], axis=0) + buy[1:237]) / adjfactor[1:237]
        
        minute_past = len(buy) - 237
        if minute_past == 1:
            new_bid = np.concatenate((new_bid, [(bid[-1] + sell[-1]) / adjfactor[-1]]))
            new_ask = np.concatenate((new_ask, [(ask[-1] + buy[-1]) / adjfactor[-1]]))
        else:
            new_bid = np.concatenate((new_bid, [(bid[-minute_past] + sell[-minute_past]) / adjfactor[-minute_past]]))
            new_bid = np.concatenate((new_bid, (np.diff(bid[-minute_past:], axis=0) + sell[(-minute_past+1):]) / adjfactor[(-minute_past+1):]))
            new_ask = np.concatenate((new_ask, [(ask[-minute_past] + buy[-minute_past]) / adjfactor[-minute_past]]))
            new_ask = np.concatenate((new_ask, (np.diff(ask[-minute_past:], axis=0) + buy[(-minute_past+1):]) / adjfactor[(-minute_past+1):]))
            
        ratio = (new_bid[-lb:] - new_ask[-lb:])
        ratio_skew = skew(ratio, axis=0, nan_policy = 'omit')
        f = np.nansum(ratio_skew * weight[-1])
 
        if np.isnan(f) or np.isinf(f):
            return 0
        else:
            return f