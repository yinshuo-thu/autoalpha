from future_factor import FutureFactor
import numpy as np


class MinuteIndexEMV(FutureFactor):
    '''
    Description: cs_mean(ts_mean(diff((high_adj + low_adj) / 2, 1) * (high_adj - low_adj) * volume_adj, 20)),
                 high_adj = high * adjfactor, low_adj = low * adjfactor, volume_adj = volume / adjfactor.
    Class: MTM
    Author: jinpx, modified by hefj
    '''
    data_type = 'IndexStock'
    days_past = 1
    data_dict = {}
    data_dict['Stock'] = ['high', 'low', 'volume', 'adjfactor']
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        lb = 20
        high = data['high'].values[-lb - 1:]
        high[high == 0] = np.nan
        low = data['low'].values[-lb - 1:]
        low[low == 0] = np.nan
        volume = data['volume'].values[-lb:]
        volume[volume == 0] = np.nan
        adj = data['adjfactor'].values[-lb - 1:]
        adj[adj == 0] = np.nan
        high = high * adj
        low = low * adj
        volume = volume / adj[-lb:]
        mid = (high + low) / 2
        mid_g = np.diff(mid, axis=0)
        hml = high[-lb:] - low[-lb:]
        emv = mid_g * hml * volume
        f = np.nanmean(np.nanmean(emv, axis=0))
        return f
