import numpy as np
from future_factor import FutureFactor

class MinuteSpotFutureCloseCorr(FutureFactor):
    '''
    Description: corr(close, close_000300.SH, 30)
    Class: Future_Spot_Price
    Author: hefj, modified by lixr
    '''
    
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Future_Data'] = ['close']
    data_dict['Index_Id'] = {'000300.SH':['close']}
    normalize_size = 5 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        lb = 30
        close = data['close'].values
        close[close == 0] = np.nan
        index_close = data['close_000300.SH'].values
        index_close[index_close == 0] = np.nan   
        mask = np.isnan(index_close[-lb:]) | np.isnan(close[-lb:])
        
        return np.corrcoef(index_close[-lb:][~mask], close[-lb:][~mask])[0,1]