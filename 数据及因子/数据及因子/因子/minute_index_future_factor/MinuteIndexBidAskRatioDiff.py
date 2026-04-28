from future_factor import FutureFactor
import numpy as np


class MinuteIndexBidAskRatioDiff(FutureFactor):
    '''
    Description: cs_weighted_mean(ts_mean(bid_ask_ratio_g, 20), w=weight),
                 bid_ask_ratio_g = diff(bid_ask_ratio, 1),
                 bid_ask_ratio = (TotalBidVol - TotalAskVol) / (TotalBidVol + TotalAskVol).
    Class: Bid_Ask
    Author: hefj
    '''
    data_type = 'IndexStock'
    days_past = 1
    data_dict = {}
    data_dict['Stock'] = ['TotalBidVol', 'TotalAskVol', 'weight']
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        bid = data['TotalBidVol'].values[-21:]
        bid[bid == 0] = np.nan
        ask = data['TotalAskVol'].values[-21:]
        ask[ask == 0] = np.nan
        weight = data['weight'].values[-1]
        ratio = (bid - ask) / (bid + ask)
        g = ratio[1:] - ratio[:-1]
        nan_num = np.isnan(g).sum(axis=0)
        g = g[:, nan_num == 0]
        f = np.nansum(np.nanmean(g, axis=0) * weight[nan_num == 0])
        return f
