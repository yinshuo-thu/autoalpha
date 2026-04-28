import numpy as np
from future_factor import FutureFactor

class MinuteIndexMoneyUniqueConsistency(FutureFactor):
    '''
    Description: 
    Class: 
    Author: jinpx
    '''    
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['BuyTradeMoney', 'SellTradeMoney', 'BuyUniqueOrderNum', 'SellUniqueOrderNum']
    normalize_size = 10 * 237
    normalize_type = 'ts_rank'

    def calculate(self, data):

        BuyTradeMoney = data['BuyTradeMoney'].values
        SellTradeMoney = data['SellTradeMoney'].values
        BuyUniqueOrderNum = data['BuyUniqueOrderNum'].values
        SellUniqueOrderNum = data['SellUniqueOrderNum'].values 
        
        N = 30
        BuyMoneyPerTrade_new = np.nansum(BuyTradeMoney[-N:], axis=0) / np.nansum(BuyUniqueOrderNum[-N:], axis=0)
        BuyMoneyPerTrade_old = np.nansum(BuyTradeMoney[-2*N:-N], axis=0) / np.nansum(BuyUniqueOrderNum[-2*N:-N], axis=0)
        BuyUniqueOrderNum_new = np.nansum(BuyUniqueOrderNum[-N:], axis=0)
        BuyUniqueOrderNum_old = np.nansum(BuyUniqueOrderNum[-2*N:-N], axis=0)
        
        up_num = np.nansum(np.logical_and(BuyMoneyPerTrade_new>BuyMoneyPerTrade_old, BuyUniqueOrderNum_new>BuyUniqueOrderNum_old))
        down_num = np.nansum(np.logical_and(BuyMoneyPerTrade_new<BuyMoneyPerTrade_old, BuyUniqueOrderNum_new<BuyUniqueOrderNum_old))
        
        f = up_num - down_num
        
        return f