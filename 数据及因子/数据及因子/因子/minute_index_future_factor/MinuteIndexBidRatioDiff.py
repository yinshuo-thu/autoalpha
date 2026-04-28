from future_factor import FutureFactor
import numpy as np


class MinuteIndexBidRatioDiff(FutureFactor):
    '''
    Description: cs_weighted_mean(ts_mean(diff(bid_ratio, 1), 30), w=weight),
                 bid_ratio[i, :] = TotalBidVol_adj[i, :] / ts_mean(TotalBidVol_adj[i - n * 237, :], n=1, 2, ..., 5),
                 TotalBidVol_adj = forward_adj(TotalBidVol).
    Class: Bid_Ask
    Author: hefj
    '''
    data_type = 'IndexStock'
    days_past = 6
    data_dict = {}
    data_dict['Stock'] = ['TotalBidVol', 'adjfactor', 'weight']
    normalize_size = 10 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        bid = data['TotalBidVol'].values[-self.days_past * 237:]
        bid[bid == 0] = np.nan
        adj = data['adjfactor'].values[-self.days_past * 237:]
        adj[adj == 0] = np.nan
        bid = bid / adj * adj[-1]
        weight = data['weight'].values[-1]
        
        bid = bid.reshape(self.days_past, 237, -1)
        bid_ratio = bid[-1] / np.nanmean(bid[:-1], axis=0)
        
        g = (bid_ratio[1:] - bid_ratio[:-1])[-30:]
        inf_nan_num = np.isnan(g).sum(axis=0) + np.isinf(g).sum(axis=0)
        g = g[:, inf_nan_num == 0]
        f = np.nansum(np.nanmean(g, axis=0) * weight[inf_nan_num == 0])
        return f
