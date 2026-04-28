from scipy import stats
import pandas as pd
import numpy as np
from future_factor import FutureFactor


class MinuteIndexUniqueSellRatioMA10(FutureFactor):
    '''
    Description: cs_mean(ts_mean(SellUniqueOrderNum, 10)) / cs_mean(ts_mean(SellTradeNum, 10))
    Class: Group_Stat
    Author: lixr, modified by shentq
    '''
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['SellUniqueOrderNum', 'SellTradeNum']
    normalize_size = 10 * 237
    normalize_type = 'ts_rank'

    def calculate(self, data):
        n = 10

        sell_unique_num = data['SellUniqueOrderNum'].values[-n:]
        sell_trade_num = data['SellTradeNum'].values[-n:]

        sell_unique_num_mean = np.nanmean(np.nanmean(sell_unique_num))
        sell_trade_num_mean = np.nanmean(np.nanmean(sell_trade_num))

        factor_value = sell_unique_num_mean / sell_trade_num_mean

        if np.isnan(factor_value) or np.isinf(factor_value):
            factor_value = 0

        return factor_value