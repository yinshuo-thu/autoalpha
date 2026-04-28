from future_factor import FutureFactor
import numpy as np


class MinuteTtimesTFutureReturnCorr(FutureFactor):
    '''
    Description: mean(rtn_t, 30) * corr(rtn, rtn_t, 240 * 5),
                 rtn = pct_chg(close, 1),
                 rtn_t = pct_chg(close_T, 1).
    Class: Treasure_Future
    Author: jinpx, modified by hefj
    '''
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 5
    data_dict = {}
    data_dict['Index_Id'] = {'000300.SH': ['close']}
    data_dict['Other_Variety'] = {'T': ['close']}
    normalize_size = 10 * 240
    normalize_type = 'ts_rank'

    def calculate(self, data):
        t = data['close_T'].values[-240 * self.days_past - 1:]
        close = data['close_000300.SH'].values[-240 * self.days_past - 1:]
        rtn_t = np.diff(t) / t[:-1]
        rtn = np.diff(close) / close[:-1]
        f = np.mean(rtn_t[-30:]) * np.corrcoef(rtn_t, rtn)[0, 1]
        return f
