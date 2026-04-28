from future_factor import FutureFactor
import numpy as np


class  MinuteContraVarietyCloseCorr(FutureFactor):
    '''
    Description: corr(Index_ClosePx, Index_Other1_ClosePx, 30)
    Class: Multi-Variety
    Author:  shentq modeified by liuz
    '''
    data_type='Future'
    instrument_type='main'
    days_past=1
    data_dict=dict()
    data_dict['Index_Id'] ={'000905.SH':['close'],'000300.SH':['close']}
    
    normalize_size=20*237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        index_close = data['close_000905.SH'].values 
        index_other_close = data['close_000300.SH'].values 

        factor= np.corrcoef(index_close[-30:],index_other_close[-30:])[0,1]
                    
        return factor
    
