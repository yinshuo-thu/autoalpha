from scipy import stats
import pandas as pd
import numpy as np
import bottleneck as bn
from future_factor import FutureFactor


class Minute60_10Speed(FutureFactor):
    '''
    Description: pct_chg(ema(ClosePx, 60), 10)
    Class: MTM
    Author: lixr, modified by shentq
    '''
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 0
    data_dict = dict()
    data_dict['Continuous_Data'] = {'IF':['close']}
    normalize_size = int(1 * 237)
    normalize_type = 'ts_rank'

    def __init__(self):
        super().__init__()
        self.ema_list = []

    def calculate(self, data):
        close = data['close_cont_IF'].values
        self.ema_list.append(self.calc_ema(self.ema_list, close[-1], 60))
        if len(self.ema_list) >= 70:
            close_ema_1 = self.ema_list[-1]
            close_ema_2 = self.ema_list[-11]
            factor_value = close_ema_1 / close_ema_2 - 1
        else:
            factor_value = 0

        return factor_value

    def rank(self, arr, axis=0, ascending=True, pct=False):
        sign = 1.0 if ascending else -1.0
        rank_value = (sign * arr).argsort(axis=axis).argsort(axis=axis) + 1
        if pct:
            rank_value = rank_value / arr.shape[axis]

        return rank_value

    def calc_ema(self, cur_ema_list, x_n, span=None):
        cur_ema_length = len(cur_ema_list)

        if span == None:
            span = cur_ema_length + 1
        else:
            span = min(cur_ema_length + 1, span)
            assert span > 0, "Invalid input of arg span!"

        alpha = 2 / (span + 1)

        if cur_ema_length == 0:
            return x_n
        else:
            return (alpha * x_n + (1 - alpha) * cur_ema_list[-1])