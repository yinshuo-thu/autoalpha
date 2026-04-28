import numpy as np
import pandas as pd
from future_factor import FutureFactor


class  MinuteMaxGrowthMaxDrawdownTimeDiff(FutureFactor):

    data_type='Future'
    instrument_type='main'
    days_past= 5
    data_dict=dict()

    data_dict['Index_Id'] = {'000300.SH':['close']}
    normalize_size= 237
    normalize_type = 'ts_rank'

    def calculate(self, data):

        lb = 237
        close_index = data['close_000300.SH'].values
        rtn = close_index[1:]/close_index[:-1]-1

        rtn = rtn[-lb:]

        i = np.argmax(np.maximum.accumulate(rtn) - rtn)  # 结束位置
        if i == 0:
            return 0
        j = np.argmax(rtn[:i])  # 开始位置

        mdd = i-j
        i = np.argmin(np.minimum.accumulate(rtn) - rtn)  # 结束位置
        if i == 0:
            return 0
        j = np.argmin(rtn[:i])  # 开始位置
        mcl = i-j

        return mcl-mdd


