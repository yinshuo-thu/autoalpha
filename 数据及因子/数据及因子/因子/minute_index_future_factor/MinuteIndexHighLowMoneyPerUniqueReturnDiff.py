import numpy as np
import bottleneck as bn
from future_factor import FutureFactor

class MinuteIndexHighLowMoneyPerUniqueReturnDiff(FutureFactor):
    '''
    Description: 
    Class: 
    Author: jinpx
    '''    
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['BuyUniqueOrderNum', 'SellUniqueOrderNum', 'BuyTradeMoney', 'SellTradeMoney', 'close', 'adjfactor']
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'

    def calculate(self, data):

        BuyUniqueOrderNum = data['BuyUniqueOrderNum'].values
        SellUniqueOrderNum = data['SellUniqueOrderNum'].values
        BuyTradeMoney = data['BuyTradeMoney'].values
        SellTradeMoney = data['SellTradeMoney'].values
        close = data['close'].values
        adjfactor = data['adjfactor'].values
        close_adj = close * adjfactor
        r = np.diff(close_adj, axis=0) / close_adj[:-1]
        
        BuyMoneyPerUnique = BuyTradeMoney / BuyUniqueOrderNum
        SellMoneyPerUnique = SellTradeMoney / SellUniqueOrderNum
        MoneyPerUnique = BuyMoneyPerUnique + SellMoneyPerUnique

        N = 60
        MoneyPerUnique_mean = np.nanmean(MoneyPerUnique[-N:], axis=0)
        MoneyPerUnique_mean[np.isnan(MoneyPerUnique_mean)] = np.nanmean(MoneyPerUnique_mean)
        MoneyPerUnique_mean_rank = (bn.rankdata(MoneyPerUnique_mean)-1)/(len(MoneyPerUnique_mean)-1)
        r_sum = np.nansum(r[-N:], axis=0)
        f = np.nanmean(r_sum[MoneyPerUnique_mean_rank>0.5]) - np.nanmean(r_sum[MoneyPerUnique_mean_rank<0.5])
        
        return f