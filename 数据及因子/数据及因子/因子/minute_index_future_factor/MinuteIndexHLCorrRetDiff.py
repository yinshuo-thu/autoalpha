import numpy as np
from future_factor import FutureFactor
import bottleneck as bn

class MinuteIndexHLCorrRetDiff(FutureFactor):
    '''
    Description: mean(return((rank(corr) > 0.5), 5days)) - mean(return((rank(corr) < 0.5), 5days))
                 corr = corr(close,close_000300.SH, 5days)
    Class: Group Stat
    Author: lixr
    '''
    
    data_type = 'IndexStock'
    days_past = 5
    data_dict = dict()
    data_dict['Stock'] = ['close', 'adjfactor']
    data_dict['Index_Id'] = {'000300.SH':['close']}
    normalize_size = 5 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        threshold = 0.5
        n = 5 * 237
        adjfactor = data['adjfactor'].values
        stock_close = data['close'].values * adjfactor
        index_close = data['close_000300.SH'].values.flatten()
        
        stock_ret = stock_close[-n:] / stock_close[-(n + 1):-1] - 1
        index_ret = index_close[-n:] / index_close[-(n + 1):-1] - 1
        corr_list = []
        for i in range(stock_close.shape[1]):
            corr = np.corrcoef(stock_ret[:,i], index_ret)[0,1]
            corr_list.append(corr)
        corr_array = np.array(corr_list)
        corr_rank = bn.rankdata(corr_array) / len(corr_array)
        
        stock_ret_mean = np.nanmean(stock_ret,axis = 0)
        factor_value = np.nanmean(stock_ret_mean[corr_rank > threshold]) - np.nanmean(stock_ret_mean[corr_rank <= (1 - threshold)])
        
        if np.isnan(factor_value) or np.isinf(factor_value):
            return 0
        else:
            return factor_value