from future_factor import FutureFactor
import numpy as np
from scipy.stats import skew


class MinuteIndexAmountRatioTSSkew(FutureFactor):
    '''
    Description: -cs_mean(ts_skew(amount_ratio, 30)),
                 amount_ratio[i, :] = amount[i, :] / ts_mean(amount[i - n * 237, :], n=1, 2, ..., 5).
    Class: Liq_Ts_Stat
    Author: hefj
    '''
    data_type = 'IndexStock'
    days_past = 6
    data_dict = {}
    data_dict['Stock'] = ['amount']
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        amount = data['amount'].values[-self.days_past * 237:].reshape(self.days_past, 237, -1)
        amount_ratio = amount[-1] / np.nanmean(amount[:-1], axis=0)
        amount_ratio = amount_ratio[-30:]
        
        nan_inf_num = np.isnan(amount_ratio).sum(axis=0) + np.isinf(amount_ratio).sum(axis=0)
        amount_ratio = amount_ratio[:, nan_inf_num == 0]
        
        f = -np.mean(skew(amount_ratio, axis=0))
        return f
