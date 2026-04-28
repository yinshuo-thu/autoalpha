from scipy import stats
import pandas as pd
import numpy as np
from future_factor import FutureFactor


class MinuteIndexUniqueBuyRatioMA10(FutureFactor):
    '''
    Description: -cs_mean(ts_mean(BuyUniqueOrderNum, 10)) / cs_mean(ts_mean(BuyTradeNum, 10))
    Class: Group_Stat
    Author: lixr, modified by shentq
    '''
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['BuyUniqueOrderNum', 'BuyTradeNum']
    normalize_size = 5 * 237
    normalize_type = 'ts_rank'

    def calculate(self, data):
        n = 10

        buy_unique_num = data['BuyUniqueOrderNum'].values[-n:]
        buy_trade_num = data['BuyTradeNum'].values[-n:]

        buy_unique_num_mean = np.nanmean(np.nanmean(buy_unique_num))
        buy_trade_num_mean = np.nanmean(np.nanmean(buy_trade_num))

        factor_value = -buy_unique_num_mean / buy_trade_num_mean

        if np.isnan(factor_value) or np.isinf(factor_value):
            factor_value = 0

        return factor_value