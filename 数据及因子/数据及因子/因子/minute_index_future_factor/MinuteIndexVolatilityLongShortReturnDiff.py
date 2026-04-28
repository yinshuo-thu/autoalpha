import numpy as np
import pandas as pd
from future_factor import FutureFactor
import bottleneck as bk

class MinuteIndexVolatilityLongShortReturnDiff(FutureFactor):
    
    data_type = 'IndexStock'
    days_past = 6
    data_dict = dict()
    data_dict['Stock'] = ['close']
    normalize_size = 237
    normalize_type = 'ts_rank'

    def calculate(self, data):

        lb_long = 237*5
        lb_short = 120
        close =  data['close'].values 

        rtn = close[1:]/close[:-1]-1
        volatility = np.nanstd(rtn[-lb_long:],axis=0)

        close_rtn_short = close[-1] / close[-lb_short] - 1

        close_rtn_long_rank =  bk.rankdata(volatility)/len(volatility)

        factor = np.nanmean(close_rtn_short[close_rtn_long_rank>=0.5])-np.nanmean(close_rtn_short[close_rtn_long_rank<=0.5])

        return factor