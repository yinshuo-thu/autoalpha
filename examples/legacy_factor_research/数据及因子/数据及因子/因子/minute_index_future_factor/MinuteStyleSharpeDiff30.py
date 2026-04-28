from future_factor import FutureFactor
import numpy as np


class MinuteStyleSharpeDiff30(FutureFactor):
    '''
    Description: mean(rtn_300 - rtn_500, 30) / std(rtn_300 - rtn_500, 30) + mean(rtn_300 - rtn_50, 30) / std(rtn_300 - rtn_50, 30),
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
    normalize_size = 10 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        lb = 31
        price_sz = data['close_000016.SH'].values[-lb:]
        price_hs = data['close_000300.SH'].values[-lb:]
        price_zz = data['close_000905.SH'].values[-lb:]
        r_sz = (price_sz[1:] - price_sz[:-1]) / price_sz[:-1]
        r_hs = (price_hs[1:] - price_hs[:-1]) / price_hs[:-1]
        r_zz = (price_zz[1:] - price_zz[:-1]) / price_zz[:-1]
        f = np.nanmean(r_hs - r_zz) / np.nanstd(r_hs - r_zz) + np.nanmean(r_hs - r_sz) / np.nanstd(r_hs - r_sz)
        return f
