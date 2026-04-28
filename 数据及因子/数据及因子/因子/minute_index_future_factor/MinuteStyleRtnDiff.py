from future_factor import FutureFactor
import numpy as np


class MinuteStyleRtnDiff(FutureFactor):
    '''
    Description: mean(rtn_300 - rtn_500, 15) + mean(rtn_300 - rtn_50),
                 rtn_300 = pct_chg((close_000300.SH + high_000300.SH + low_000300.SH) / 3, 1),
                 rtn_500 = pct_chg((close_000905.SH + high_000905.SH + low_000905.SH) / 3, 1),
                 rtn_50 = pct_chg((close_000016.SH + high_000016.SH + low_000016.SH) / 3, 1).
    Class: Multi-Variety
    Author: hefj
    '''
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = {}
    data_dict['Index_Id'] = {'000016.SH': ['close', 'high', 'low'], '000300.SH': ['close', 'high', 'low'], '000905.SH': ['close', 'high', 'low']}
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        price_sz = (data['close_000016.SH'].values[-16:] + data['high_000016.SH'].values[-16:] + data['low_000016.SH'].values[-16:]) / 3
        price_hs = (data['close_000300.SH'].values[-16:] + data['high_000300.SH'].values[-16:] + data['low_000300.SH'].values[-16:]) / 3
        price_zz = (data['close_000905.SH'].values[-16:] + data['high_000905.SH'].values[-16:] + data['low_000905.SH'].values[-16:]) / 3
        r_sz = (price_sz[1:] - price_sz[:-1]) / price_sz[:-1]
        r_hs = (price_hs[1:] - price_hs[:-1]) / price_hs[:-1]
        r_zz = (price_zz[1:] - price_zz[:-1]) / price_zz[:-1]
        f = np.nanmean(r_hs - r_zz) + np.nanmean(r_hs - r_sz)
        return f
