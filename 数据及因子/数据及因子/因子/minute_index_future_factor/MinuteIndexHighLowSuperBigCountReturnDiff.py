import numpy as np
import bottleneck as bn
from future_factor import FutureFactor

class MinuteIndexHighLowSuperBigCountReturnDiff(FutureFactor):
    '''
    Description: 
    Class: Group_Stat
    Author: jinpx
    '''    
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['buy_superorder_count', 'sell_superorder_count', 'buy_bigorder_count', 'sell_bigorder_count',
                          'buy_midorder_count', 'sell_midorder_count', 'buy_smallorder_count', 'sell_smallorder_count',
                          'close', 'adjfactor']
    normalize_size = 180
    normalize_type = 'ts_rank'

    def calculate(self, data):
        
        buy_superorder_count = data['buy_superorder_count'].values
        sell_superorder_count = data['sell_superorder_count'].values
        buy_bigorder_count = data['buy_bigorder_count'].values
        sell_bigorder_count = data['sell_bigorder_count'].values
        buy_midorder_count = data['buy_midorder_count'].values
        sell_midorder_count = data['sell_midorder_count'].values
        buy_smallorder_count = data['buy_smallorder_count'].values
        sell_smallorder_count = data['sell_smallorder_count'].values
        close = data['close'].values
        adjfactor = data['adjfactor'].values
        close_adj = close * adjfactor
        r = np.diff(close_adj, axis=0) / close_adj[:-1]

        N = 5*237
        buy_super_big_count_ratio = (buy_superorder_count + buy_bigorder_count) / (buy_superorder_count + buy_bigorder_count + buy_midorder_count + buy_smallorder_count)
        sell_super_big_count_ratio = (sell_superorder_count + sell_bigorder_count) / (sell_superorder_count + sell_bigorder_count + sell_midorder_count + sell_smallorder_count)
        buy_sell_super_big_count_ratio_sum = np.nanmean(buy_super_big_count_ratio[-N:], axis=0) + np.nanmean(sell_super_big_count_ratio[-N:], axis=0)
        buy_sell_super_big_count_ratio_sum_rank = (bn.rankdata(buy_sell_super_big_count_ratio_sum)-1)/(len(buy_sell_super_big_count_ratio_sum)-1)
        r_sum = np.nansum(r[-N:], axis=0)
        f = -(np.nanmean(r_sum[buy_sell_super_big_count_ratio_sum_rank>0.8]) - np.nanmean(r_sum[buy_sell_super_big_count_ratio_sum_rank<0.2]))
   
        return f