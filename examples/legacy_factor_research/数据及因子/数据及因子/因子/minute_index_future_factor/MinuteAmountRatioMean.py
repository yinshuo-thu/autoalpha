from future_factor import FutureFactor
import numpy as np


class MinuteAmountRatioMean(FutureFactor):
    '''
    Description: mean(amount_ratio, 20),
                 amount_ratio[i] = amount_000300.SH[i] / mean(amount_000300.SH[i - n * 237], n=1, 2, ..., 5).
    Class: Liquidity
    Author: hefj
    '''
    data_type = 'Future'
    days_past = 6
    data_dict = {}
    data_dict['Index_Id'] = {'000300.SH': ['amount']}
    normalize_size = 10 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        amount = data['amount_000300.SH'].values[-self.days_past * 237:].reshape(self.days_past, -1)
        amount_ratio = amount[-1] / np.nanmean(amount[:-1], axis=0)
        f = np.mean(amount_ratio[-20:])
        return f
