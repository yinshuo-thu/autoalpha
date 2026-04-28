import numpy as np
import bottleneck as bn
from future_factor import FutureFactor

class MinuteIndexHighLowBuySellTradeNumSumReturnDiff(FutureFactor):
    '''
    Description: 
    Class: Group_Stat
    Author: jinpx
    '''    
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 5
    data_dict = dict()
    data_dict['Stock'] = ['BuyTradeNum', 'SellTradeNum', 'close', 'adjfactor']
    normalize_size = 150
    normalize_type = 'ts_rank'

    def calculate(self, data):
        
        BuyTradeNum = data['BuyTradeNum'].values
        SellTradeNum = data['SellTradeNum'].values
        close = data['close'].values
        adjfactor = data['adjfactor'].values
        close_adj = close * adjfactor
        r = np.diff(close_adj, axis=0) / close_adj[:-1]
        
        BuySellTradeNumSum = BuyTradeNum + SellTradeNum
        N = 5 * 237
        BuySellTradeNumSum_mean = np.nanmean(BuySellTradeNumSum[-N:], axis=0)
        BuySellTradeNumSum_mean_rank = (bn.rankdata(BuySellTradeNumSum_mean)-1)/(len(BuySellTradeNumSum_mean)-1)
        r_sum = np.nansum(r[-N:], axis=0)
        f = np.nanmean(r_sum[BuySellTradeNumSum_mean_rank>0.8]) - np.nanmean(r_sum[BuySellTradeNumSum_mean_rank<0.2])
        
        return f