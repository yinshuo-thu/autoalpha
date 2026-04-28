import numpy as np
import bottleneck as bn
from future_factor import FutureFactor

class MinuteIndexHighLowBuySellUniqueOrderNumSumReturnDiff(FutureFactor):
    '''
    Description: 
    Class: Group_Stat
    Author: jinpx
    '''    
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['BuyUniqueOrderNum', 'SellUniqueOrderNum', 'close', 'adjfactor']
    normalize_size = 180
    normalize_type = 'ts_rank'

    def calculate(self, data):
        
        BuyUniqueOrderNum = data['BuyUniqueOrderNum'].values
        SellUniqueOrderNum = data['SellUniqueOrderNum'].values
        close = data['close'].values
        adjfactor = data['adjfactor'].values
        close_adj = close * adjfactor
        r = np.diff(close_adj, axis=0) / close_adj[:-1]
        
        BuySellUniqueOrderNumSum = BuyUniqueOrderNum + SellUniqueOrderNum
        N = 1 * 237
        BuySellUniqueOrderNumSum_mean = np.nanmean(BuySellUniqueOrderNumSum[-N:], axis=0)
        BuySellUniqueOrderNumSum_mean_rank = (bn.rankdata(BuySellUniqueOrderNumSum_mean)-1)/(len(BuySellUniqueOrderNumSum_mean)-1)
        r_sum = np.nansum(r[-N:], axis=0)
        f = np.nanmean(r_sum[BuySellUniqueOrderNumSum_mean_rank>0.8]) - np.nanmean(r_sum[BuySellUniqueOrderNumSum_mean_rank<0.2])
        
        return f