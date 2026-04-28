import numpy as np
import bottleneck as bk
from future_factor import FutureFactor

class MinuteIndexHLTurnoverBSBigOrderCountRatioDiff(FutureFactor):
    '''
    Description: mean(ratio(rank > 0.9)) - mean(ratio(rank < 0.1)),
                 ratio = mean(buy_bigorder_count / sell_bigorder_count, 60)
                 rank = rank(mean(turnover_rate, 237))
    Class: Group_Stat
    Author: lixr
    '''
    
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['buy_bigorder_count','sell_bigorder_count','turnover_rate']
    normalize_size = 10 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        n1 = 60
        n2 = 237
        threshold = 0.9
        
        buy = data['buy_bigorder_count'].values
        buy[buy == 0] = np.nan
        sell = data['sell_bigorder_count'].values
        sell[sell == 0] = np.nan
        turnover = data['turnover_rate'].values
        
        ratio = np.nanmean(buy[-n1:] / sell[-n1:], axis = 0)
        mask = np.isnan(ratio)
        ratio = ratio[~mask]
        turnover_mean = np.nanmean(turnover[-n2:], axis = 0)[~mask]
        rank = bk.rankdata(turnover_mean) / len(turnover_mean)
        factor_value = np.nanmean(ratio[rank > threshold]) - np.nanmean(ratio[rank <= (1 - threshold)])
        
        if np.isnan(factor_value) or np.isinf(factor_value):
            return 0
        else:
            return factor_value