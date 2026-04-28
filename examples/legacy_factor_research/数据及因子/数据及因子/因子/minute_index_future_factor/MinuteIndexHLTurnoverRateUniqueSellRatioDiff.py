import numpy as np
import bottleneck as bk
from future_factor import FutureFactor

class MinuteIndexHLTurnoverRateUniqueSellRatioDiff(FutureFactor):
    '''
    Description: mean(ratio(rank > 0.8)) - mean(ratio(rank < 0.2)),
                 ratio = -1 * mean(SellTradeNum / SellUniqueOrderNum, 20)
                 rank = rank(mean(turnover_rate, 60))
    Class: Group_Stat
    Author: lixr
    '''
    
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['SellUniqueOrderNum','SellTradeNum','turnover_rate']
    normalize_size = 10 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        n1 = 20
        n2 = 60
        threshold = 0.8
        
        sell_unique = data['SellUniqueOrderNum'].values
        sell_unique[sell_unique == 0] = np.nan
        sell = data['SellTradeNum'].values
        sell[sell == 0] = np.nan
        turnover = data['turnover_rate'].values
        
        ratio = -1 * np.nanmean(sell[-n1:] / sell_unique[-n1:], axis = 0)
        mask = np.isnan(ratio)
        ratio = ratio[~mask]
        turnover_mean = np.nanmean(turnover[-n2:], axis = 0)[~mask]
        rank = bk.rankdata(turnover_mean) / len(turnover_mean)
        factor_value = np.nanmean(ratio[rank > threshold]) - np.nanmean(ratio[rank <= (1 - threshold)])
        
        if np.isnan(factor_value) or np.isinf(factor_value):
            return 0
        else:
            return factor_value