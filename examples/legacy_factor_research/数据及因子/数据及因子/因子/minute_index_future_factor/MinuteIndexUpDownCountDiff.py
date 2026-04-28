from future_factor import FutureFactor
import numpy as np


class  MinuteIndexUpDownCountDiff(FutureFactor):
    '''
    Description: "(cs_sum(where(ClosePx * Adjfactor / shift(ClosePx * Adjfactor, 60) > Index_ClosePx / shift(Index_ClosePx, 60), 1, 0))
        - cs_sum(where(ClosePx * Adjfactor / shift(ClosePx * Adjfactor, 60) < Index_ClosePx / shift(Index_ClosePx, 60), 1, 0)))[-1]"
    Class: Group_Stat
    Author:  shentq  modeified by liuz
    '''
    data_type = 'IndexStock'
    instrument_type='main'
    days_past=1
    data_dict=dict()
    data_dict['Stock'] = ['close', 'adjfactor']
    data_dict['Index_Id'] = {'000300.SH':['close']}
    
    normalize_size=20*237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):

        rtn_back = 120

        index_close = data['close_000300.SH'].values
        close = data['close'].values
        adjfactor = data['adjfactor'].values

        close_adj = close*adjfactor
        rtn = close_adj[rtn_back:]/close_adj[:-rtn_back]-1
        rtn_index = index_close[rtn_back:]/index_close[:-rtn_back]-1
        
        upcount = np.nansum(np.where(rtn[-1]> rtn_index[-1], 1,0))
        downcount =  np.nansum(np.where(rtn[-1] < rtn_index[-1], 1,0))
        
        factor = downcount-upcount
        
        return  factor
