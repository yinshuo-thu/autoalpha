import numpy as np
from future_factor import FutureFactor

class MinuteIndexBuySellRatioSharpe(FutureFactor):
    '''
    Description: ts_mean(weighted_cs_mean(BuyTradeQuantity / SellTradeQuantity - 1, w=index_weight), 30)
                / ts_std(weighted_cs_mean(BuyTradeQuantity / SellTradeQuantity - 1, w=index_weight), 30)
    Class: Buy_Sell
    Author: hefj, modified by lixr
    '''
    
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['weight','BuyTradeQuantity', 'SellTradeQuantity']
    normalize_size = 5 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        lb = 30
        weight = data['weight'].values
        buy = data['BuyTradeQuantity'].values
        sell = data['SellTradeQuantity'].values
        
        ratio = buy[-lb:] / sell[-lb:] - 1
        ratio[np.isinf(ratio)] = np.nan
        ratio = np.nansum(ratio * weight[-lb:], axis=1)
        f = np.nanmean(ratio) / np.nanstd(ratio)
        
        if np.isnan(f) or np.isinf(f):
            return 0
        else:
            return f