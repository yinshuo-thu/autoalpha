import numpy as np
from future_factor import FutureFactor

class MinuteBounceUpStdRatio(FutureFactor):
    '''
    Description: max((close_000300.SH - cum_min(low_000300.SH)) / cum_min(low_000300.SH), 30) /
                 std(where(close_000300.SH > delay(close_000300.SH, 1), pct_chg(close_000300.SH, 1), 30))
    Class: Return_Risk
    Author: hefj, modified by lixr
    '''
    
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Index_Id'] = {'000300.SH':['close','low']}
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        lb = 30
        close = data['close_000300.SH'].values
        close[close == 0] = np.nan
        close_temp = close[-lb:]
        low = data['low_000300.SH'].values
        low[low == 0] = np.nan
        low_temp = low[-lb:]
        mask = np.isnan(close_temp) | np.isnan(low_temp)
        close_temp = close_temp[~mask]
        low_temp = low_temp[~mask]
        
        cum_min = np.minimum.accumulate(low_temp)
        mbb = ((close_temp - cum_min) / cum_min).max()
        r = close_temp[1:] / close_temp[:-1] - 1
        up_std = r[r > 0].std()
        if up_std == 0 or np.isnan(up_std):
            up_ratio = 0
        else:
            up_ratio = mbb / up_std
            
        return up_ratio