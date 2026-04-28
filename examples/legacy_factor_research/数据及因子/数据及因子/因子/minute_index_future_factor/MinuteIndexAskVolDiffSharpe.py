from future_factor import FutureFactor
import numpy as np


class MinuteIndexAskVolDiffSharpe(FutureFactor):
    '''
    Description: -cs_mean(ts_mean(ask_g, 20) / ts_std(ask_g, 20)),
                 ask_g = diff(TotalAskVol / adjfactor * adjfactor[-1], 1)
    Class: Bid_Ask
    Author: hefj
    '''
    data_type = 'IndexStock'
    days_past = 1
    data_dict = {}
    data_dict['Stock'] = ['TotalAskVol', 'adjfactor']
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        ask = data['TotalAskVol'].values[-21:]
        ask[ask == 0] = np.nan
        adj = data['adjfactor'].values[-21:]
        adj[adj == 0] = np.nan
        ask = ask / adj * adj[-1]
        ask_g = ask[1:] - ask[:-1]
        nan_num = np.isnan(ask_g).sum(axis=0)
        ask_g = ask_g[:, nan_num < 5]
        f = -np.nanmean(np.nanmean(ask_g, axis=0) / np.nanstd(ask_g, axis=0))
        return f
