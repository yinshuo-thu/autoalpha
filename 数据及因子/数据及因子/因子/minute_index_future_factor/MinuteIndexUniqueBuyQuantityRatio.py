from scipy import stats
import pandas as pd
import numpy as np
from future_factor import FutureFactor


class MinuteIndexUniqueBuyQuantityRatio(FutureFactor):
    '''
    Description:
    Class: Group_Stat
    Author: lixr, modified by shentq
    '''
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['BuyUniqueOrderNum', 'BuyTradeQuantity']
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'

    def calculate(self, data):
        n1 = 240
        n2 = 10

        buy_unique_num = data['BuyUniqueOrderNum'].values[-n1:]
        buy_trade_quantity = data['BuyTradeQuantity'].values[-n1:]

        buy_order_size = buy_trade_quantity / buy_unique_num
        buy_order_size = np.where(np.isinf(buy_order_size), np.nan, buy_order_size)

        factor_value = np.nanmean(np.nanmean(buy_order_size[-n2:] / np.nanmean(buy_order_size, axis=0)))

        if np.isnan(factor_value) or np.isinf(factor_value):
            factor_value = 0

        return factor_value