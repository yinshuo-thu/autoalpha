import numpy as np
import bottleneck as bk
from future_factor import FutureFactor

class MinuteIndexHLBigSmallOrderCountRatioRetDiff(FutureFactor):
    '''
    Description: mean(return((rank(ratio) > 0.8), 5days)) - mean(return((rank(ratio) < 0.2), 5days)),
                 ratio = mean(buy_smallorder_count + sell_smallorder_count, 5days) / mean(buy_bigorder_count + sell_bigorder_count, 5days)           
    Class: Group Stat
    Author: lixr
    '''
    
    data_type = 'IndexStock'
    days_past = 5 
    data_dict = dict()
    data_dict['Stock'] = ['close','adjfactor','weight', \
                          'buy_smallorder_count','sell_smallorder_count','buy_bigorder_count','sell_bigorder_count']
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        threshold = 0.8
        n1 = 5 * 237
        n2 = 5 * 237
        adjfactor = data['adjfactor'].values
        weight = data['weight'].values[-1]
        close = data['close'].values * adjfactor
        close[close == 0] = np.nan
        small_buy = data['buy_smallorder_count'].values
        small_buy[small_buy == 0] = np.nan
        small_sell = data['sell_smallorder_count'].values
        small_sell[small_sell == 0] = np.nan
        big_buy = data['buy_bigorder_count'].values
        big_buy[big_buy == 0] = np.nan
        big_sell = data['sell_bigorder_count'].values
        big_sell[big_sell == 0] = np.nan
        
        small = np.nanmean(small_buy[-n1:], axis = 0) + np.nanmean(small_sell[-n1:], axis = 0)
        big = np.nanmean(big_buy[-n1:], axis = 0) + np.nanmean(big_sell[-n1:], axis = 0)
        ratio = small / big
        rank = bk.rankdata(ratio) / len(ratio)
        ret = (close[-1] / close[-(n2 + 1)] - 1) * weight 
        factor_value = np.nanmean(ret[rank > threshold]) - np.nanmean(ret[rank <= (1 - threshold)])
        
        if np.isnan(factor_value) or np.isinf(factor_value):
            return 0
        else:
            return factor_value