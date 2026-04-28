import numpy as np
from future_factor import FutureFactor

class MinuteIndexTradeMoneyRatioDiff(FutureFactor):
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['BuyTradeMoney', 'SellTradeMoney', 'AskP0', 'AskV0', 'BidP0', 'BidV0']
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'

    def calculate(self, data):
        n = 30

        buy_trade_money = data['BuyTradeMoney'].values
        sell_trade_money = data['SellTradeMoney'].values
        ask_amt = data['AskP0'].values * data['AskV0'].values
        bid_amt = data['BidP0'].values * data['BidV0'].values

        buy_sell_ratio = buy_trade_money / ask_amt - sell_trade_money / bid_amt
        buy_sell_ratio[np.isinf(buy_sell_ratio)] = np.nan

        factor_value = np.nanmean(np.nanmean(buy_sell_ratio[-n:]))

        return factor_value