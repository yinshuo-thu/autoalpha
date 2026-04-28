import numpy as np
from future_factor import FutureFactor

class MinuteIndexBuySellRatioAutoCorrSharpe(FutureFactor):
    '''
    Description: mean(autocorr(bs_ratio)) / std(autocorr(bs_ratio)),
                 where bs_ratio = ((BuyTradeMoney - SellTradeMoney) / (BuyTradeMoney + SellTradeMoney),30)     
    Class: Buy_Sell
    Author: hefj, modified by lixr
    '''
    
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['weight','BuyTradeMoney', 'SellTradeMoney']
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        lb = 30
        weight = data['weight'].values
        buy_money = data['BuyTradeMoney'].values
        sell_money = data['SellTradeMoney'].values
        
        bs_ratio = (buy_money[-lb:] - sell_money[-lb:]) / (buy_money[-lb:] + sell_money[-lb:])
        auto_corr = np.nanmean((bs_ratio[:-1] - np.nanmean(bs_ratio, axis=0)) * (bs_ratio[1:] - np.nanmean(bs_ratio, axis=0)), axis=0) / np.nanvar(bs_ratio, axis=0)
        mean = np.nansum(auto_corr * weight[-1])
        std = np.nansum(((auto_corr - mean) ** 2) * weight[-1]) ** 0.5
        f = mean / std
        
        if np.isnan(f) or np.isinf(f):
            return 0
        else:
            return f