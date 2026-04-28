from future_factor import FutureFactor
import numpy as np


class MinuteIndexHighLowBetaRtnDiff(FutureFactor):
    data_type = 'IndexStock'
    days_past = 5
    data_dict = {}
    data_dict['Stock'] = ['close', 'adjfactor']
    data_dict['Index_Id'] = {'000300.SH': ['close']}
    normalize_size = 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        lb = 5 * 237
        adj = data['adjfactor'].values
        close = (data['close'].values * adj / adj[-1])[-lb - 1:]
        rtn = close[1:] / close[:-1] - 1
        zero_num = (np.abs(rtn) < 1 / 10000).sum(axis=0)
        rtn = rtn[:, zero_num < (lb / 2)]
        
        index_close = data['close_000300.SH'].values[-lb - 1:]
        index_close = index_close.reshape(len(index_close), -1)
        index_rtn = index_close[1:] / index_close[:-1] - 1
        
        beta = np.mean((rtn - np.mean(rtn, axis=0)) * (index_rtn - np.mean(index_rtn)), axis=0) / np.var(index_rtn)
        median = np.nanmedian(beta)
        
        rtn_high = rtn[:, beta > median]
        rtn_low = rtn[:, beta <= median]
        
        f = np.nanmean(rtn_high) - np.nanmean(rtn_low)
        return f
