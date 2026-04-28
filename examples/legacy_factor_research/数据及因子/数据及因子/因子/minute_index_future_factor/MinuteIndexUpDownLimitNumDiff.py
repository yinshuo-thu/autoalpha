from future_factor import FutureFactor
import numpy as np


class MinuteIndexUpDownLimitNumDiff(FutureFactor):
    data_type = 'IndexStock'
    days_past = 1
    data_dict = {}
    data_dict['Stock'] = ['TotalBidVol', 'TotalAskVol']
    normalize_size = 60
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        ask = data['TotalAskVol'].values[-1]
        bid = data['TotalBidVol'].values[-1]
        up_limit_num = ((bid > 0) & (ask == 0)).sum()
        down_limit_num = ((bid == 0) & (ask > 0)).sum()
        f = up_limit_num - down_limit_num
        return f
