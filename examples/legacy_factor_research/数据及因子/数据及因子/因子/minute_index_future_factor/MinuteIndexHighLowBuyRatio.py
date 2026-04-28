import numpy as np
from future_factor import FutureFactor
import bottleneck as bn

class MinuteIndexHighLowBuyRatio(FutureFactor):
    '''
    Description: MA(ts_mean(where(close_000905.SH > quantile(close_000905.SH, 1/2, 30), buyratio, nan), 30)
                / ts_mean(where(close_000905.SH < quantile(close_000905.SH, 1/2, 30), buyratio, nan), 30), 5),
                buyratio = cs_mean(BuyTradeMoney / (BuyTradeMoney + SellTradeMoney))
    Class: Buy_Sell
    Author: hefj, modified by lixr
    '''
    
    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['BuyTradeMoney', 'SellTradeMoney']
    data_dict['Index_Id'] = {'000905.SH':['close']}
    normalize_size = 30 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        lb = 30
        buy = data['BuyTradeMoney']
        close = data['close_000905.SH'].reindex(index = buy.index).values.flatten()
        close[close == 0] = np.nan
        buy = buy.values
        sell = data['SellTradeMoney'].values
        
        minute_past = min(len(buy) - 237, 5)
        factor_temp_list = []
        for i in range(minute_past):
            if i == 0:
                buy_sell_ratio = np.nanmean(buy[-lb:] / (buy[-lb:] + sell[-lb:]), axis=1)
                close_temp = close[-lb:]
            else:
                buy_sell_ratio = np.nanmean(buy[-(lb + i):-i] / (buy[-(lb + i):-i] + sell[-(lb + i):-i]), axis=1)
                close_temp = close[-(lb + i):-i]
            rank = bn.rankdata(close_temp)
            
            f = buy_sell_ratio[rank<=15].mean() / buy_sell_ratio[rank>15].mean()
            
            if np.isnan(f) or np.isinf(f):
                factor_temp_list.append(1)
            else:
                factor_temp_list.append(f)
                
        return np.mean(factor_temp_list)