from future_factor import FutureFactor
import numpy as np


class  MinuteFSTurnRatioCloseCorr(FutureFactor):
    '''
    Description: "-corr(ClosePx, index_turnover_ratio_all, 60),
                index_turnover_ratio_all = (Contract0_Turnover + Contract1_Turnover + Contract2_Turnover + Contract3_Turnover) / Index_Turnover"
    Class:PV_Corr
    Author: shentq modeified by liuz
    '''
    data_type='Future'
    instrument_type='main'
    days_past=1
    data_dict=dict()

    data_dict['Future_Data'] = ['close']
    data_dict['Index_Id'] = {'000300.SH':['amount',]}
    data_dict['Other_Future_Instrument'] = {'00':['amount'],'01':['amount'],'02':['amount'],'03':['amount']}
    
    normalize_size= 20 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        index_turnover = data['amount_000300.SH'].values
        future_turnover = [data['amount_00'].values[i] + data['amount_01'].values[i] + data['amount_02'].values[i] + data['amount_03'].values[i] for i in range(len(data['amount_02'].values))] 
        index_close =  data['close'].values
        turnover_ratio = []
        for i in range(len(index_turnover)):
            if index_turnover[i] == 0:
                turnover_ratio.append(1)
            else:
                turnover_ratio.append(future_turnover[i] / index_turnover[i])
        factor = -np.corrcoef(turnover_ratio[-90:],index_close[-90:])[0,1]

        return factor
