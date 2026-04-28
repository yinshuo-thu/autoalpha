import numpy as np
from future_factor import FutureFactor

class MinuteCloseTurnoverRatioCorrAbs(FutureFactor):
    '''
    Description: abs(corr(close_000300.SH, amount_ratio, 60)),
                 amount_ratio =  / (the average amount_000300.SH at current time over past 5 trading days)
    Class: PV_Corr
    Author: hefj, modified by lixr
    '''
    
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 6
    data_dict = dict()
    data_dict['Index_Id'] = {'000300.SH':['close','amount']}
    normalize_size = 5 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        close = data['close_000300.SH'].values
        close[close == 0] = np.nan
        turn = data['amount_000300.SH'].values
        past_turn = turn[:237 * self.days_past].reshape(self.days_past, -1)
        past_turn_ratio = past_turn[-1] / np.nanmean(past_turn[:-1], axis=0)
        turn_mean = np.nanmean(past_turn[1:], axis=0)
        
        lb = 60
        today_turn = turn[237 * self.days_past:]
        turn_ratio = np.concatenate((past_turn_ratio, today_turn / turn_mean[:len(today_turn)]))
        close_temp = close[-lb:]
        turn_ratio_temp = turn_ratio[-lb:]
        mask = np.isnan(close_temp) | np.isnan(turn_ratio_temp)
        f = np.abs(np.corrcoef(close_temp[~mask], turn_ratio_temp[~mask])[0,1])
            
        if np.isnan(f) or np.isinf(f):
            return 0
        else:
            return f