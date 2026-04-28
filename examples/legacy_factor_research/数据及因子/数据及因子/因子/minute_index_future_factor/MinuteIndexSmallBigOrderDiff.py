import numpy as np
import pandas as pd
from future_factor import FutureFactor

class MinuteIndexSmallBigOrderDiff(FutureFactor):

    
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['buy_smallorder_money', 'buy_bigorder_money' ,'sell_smallorder_money', 'sell_bigorder_money']
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        lb = 30
        buy_smallorder_money =  data['buy_smallorder_money'].values 
        buy_bigorder_money =  data['buy_bigorder_money'].values 
        sell_smallorder_money =  data['sell_smallorder_money'].values 
        sell_bigorder_money =  data['sell_bigorder_money'].values 

        buy = np.nansum(buy_smallorder_money[-lb:],axis=0)/np.nansum(buy_bigorder_money[-lb:],axis=0)
        buy[np.isinf(buy)] = np.nan
        sell = np.nansum(sell_smallorder_money[-lb:],axis=0)/np.nansum(sell_bigorder_money[-lb:],axis=0)
        sell[np.isinf(sell)] = np.nan
        factor = np.nanmean(buy)-np.nanmean(sell)
        return -factor