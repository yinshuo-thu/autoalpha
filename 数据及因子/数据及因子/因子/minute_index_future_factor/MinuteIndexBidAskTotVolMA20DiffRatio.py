from scipy import stats
import pandas as pd
import numpy as np
from future_factor import FutureFactor


class MinuteIndexBidAskTotVolMA20DiffRatio(FutureFactor):
    '''
    Description: ts_mean((cs_sum(TotalBidVol) - cs_sum(TotalAskVol)) / (cs_sum(TotalBidVol) + cs_sum(TotalAskVol)), 20)
    Class: Bid_Ask
    Author: lixr, modified by shentq
    '''
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['TotalBidVol', 'TotalAskVol', 'adjfactor']
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'

    def calculate(self, data):
        n = 20

        total_bid = np.nansum(data['TotalBidVol'].values[-n:] / data['adjfactor'].values[-n:], axis=1)
        total_ask = np.nansum(data['TotalAskVol'].values[-n:] / data['adjfactor'].values[-n:], axis=1)

        bid_mean = np.nanmean(total_bid)
        ask_mean = np.nanmean(total_ask)

        if (bid_mean + ask_mean) == 0:
            factor_value = 0
        else:
            factor_value = (bid_mean - ask_mean) / (bid_mean + ask_mean)

        return factor_value