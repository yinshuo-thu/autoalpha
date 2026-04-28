from future_factor import FutureFactor
import numpy as np


class MinuteStyleSharpeDiff237(FutureFactor):
    '''
    Description: 2 * mean(rtn_300, 237) / std(rtn_300, 237) - mean(rtn_500, 237) / std(rtn_500, 237) - mean(rtn_50, 237) / std(rtn_50, 237),
                 rtn_300 = pct_chg(close_000300.SH, 1),
                 rtn_500 = pct_chg(close_000905.SH, 1),
                 rtn_50 = pct_chg(close_000016.SH, 1).
    Class: Multi-Variety
    Author: hefj
    '''
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = {}
    data_dict['Index_Id'] = {'000016.SH': ['close'], '000300.SH': ['close'], '000905.SH': ['close']}
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        lb = 238
        price_sz = data['close_000016.SH'].values[-lb:]
        price_hs = data['close_000300.SH'].values[-lb:]
        price_zz = data['close_000905.SH'].values[-lb:]
        r_sz = (price_sz[1:] - price_sz[:-1]) / price_sz[:-1]
        r_hs = (price_hs[1:] - price_hs[:-1]) / price_hs[:-1]
        r_zz = (price_zz[1:] - price_zz[:-1]) / price_zz[:-1]
        f = 2 * np.nanmean(r_hs) / np.nanstd(r_hs) - np.nanmean(r_zz) / np.nanstd(r_zz) - np.nanmean(r_sz) / np.nanstd(r_sz)
        return f
