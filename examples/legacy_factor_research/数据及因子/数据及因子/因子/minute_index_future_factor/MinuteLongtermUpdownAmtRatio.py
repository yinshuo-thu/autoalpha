from future_factor import FutureFactor
import numpy as np


class  MinuteLongtermUpdownAmtRatio(FutureFactor):
    '''
    Description: "-corr(ClosePx, index_turnover_ratio_all, 60),
                index_turnover_ratio_all = (Contract0_Turnover + Contract1_Turnover + Contract2_Turnover + Contract3_Turnover) / Index_Turnover"
    Class:PV_Corr
    Author: shentq modeified by liuz
    '''
    data_type='Future'
    instrument_type='main'
    days_past= 5
    data_dict=dict()


    data_dict['Index_Id'] = {'000300.SH':['close','amount']}
    normalize_size= 60
    normalize_type = 'ts_rank'

    def calculate(self, data):
        lb = 237*5

        close_index = data['close_000300.SH'].values[-lb:]
        amount_index = data['amount_000300.SH'].values[-lb:][1:]
        re = close_index[1:]/close_index[:-1]-1

        factor = np.nansum(amount_index[re>0])/np.nansum(amount_index[re<=0])
        return factor

