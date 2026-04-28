import numpy as np
from future_factor import FutureFactor

class MinuteIndexObLv1AmtPerOrderRatioSharpe(FutureFactor):
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['Bid1AmtMean','Ask1AmtMean','Buy1NumOrdersMean','Sell1NumOrdersMean']
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'

    def calculate(self, data):
        n = 30

        bid_amt = data['Bid1AmtMean'].values
        ask_amt = data['Ask1AmtMean'].values

        bid_order_num = data['Buy1NumOrdersMean'].values
        ask_order_num = data['Sell1NumOrdersMean'].values

        bid_amt_per_order = bid_amt / bid_order_num
        ask_amt_per_order = ask_amt / ask_order_num

        order_amt_ratio = (bid_amt_per_order - ask_amt_per_order) / (bid_amt_per_order + ask_amt_per_order)

        factor_value = np.nanmean(np.nanmean(order_amt_ratio[-n:],axis=1)) / np.nanstd(np.nanmean(order_amt_ratio[-n:],axis=1))

        if np.isnan(factor_value) or np.isinf(factor_value):
            factor_value = 0

        return factor_value