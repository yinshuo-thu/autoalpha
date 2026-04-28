from future_factor import FutureFactor
import numpy as np


class MinuteIndexHighLowSpreadReturnDiff(FutureFactor):
    '''
    Description: ts_mean(cs_mean(rtn_high), 45) - ts_mean(cs_mean(rtn_low), 45),
                 rtn_high = pct_chg(close * adjfactor, 1)[:, spread > quantile(spread, 0.8)],
                 rtn_low = pct_chg(close * adjfactor, 1)[:, spread < quantile(spread, 0.2)],
                 spread = ts_mean((BidP0 - AskP0) / ((BidP0 + AskP0) / 2), from 09:30 T-1).
    Class: Group_Stat
    Author: jinpx, modified by hefj
    '''
    data_type = 'IndexStock'
    days_past = 1
    data_dict = {}
    data_dict['Stock'] = ['close', 'AskP0', 'BidP0', 'adjfactor']
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        close = data['close'].values[-46:]
        close[close == 0] = np.nan
        adj = data['adjfactor'].values[-46:]
        adj[adj == 0] = np.nan
        close = close * adj
        rtn = np.diff(close, axis=0) / close[:-1]
        ask = data['AskP0'].values
        ask[ask == 0] = np.nan
        bid = data['BidP0'].values
        bid[bid == 0] = np.nan
        spread = np.nanmean((bid - ask) / ((bid + ask) / 2), axis=0)
        high_level = np.quantile(spread[~np.isnan(spread)], 0.5)
        low_level = np.quantile(spread[~np.isnan(spread)], 0.5)
        rtn_high = np.nanmean(rtn[:, spread > high_level], axis=1)
        rtn_low = np.nanmean(rtn[:, spread < low_level], axis=1)
        f = np.nanmean(rtn_high) - np.nanmean(rtn_low)
        return f
