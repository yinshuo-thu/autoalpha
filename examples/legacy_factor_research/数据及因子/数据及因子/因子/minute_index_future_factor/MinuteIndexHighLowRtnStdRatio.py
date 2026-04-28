from future_factor import FutureFactor
import numpy as np
from scipy.stats import pearsonr


class MinuteIndexHighLowRtnStdRatio(FutureFactor):
    data_type = 'IndexStock'
    days_past = 1
    data_dict = {}
    data_dict['Stock'] = ['close', 'adjfactor']
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        lb = 60
        adj = data['adjfactor'].values[-lb:]
        close = data['close'].values[-lb:] * adj / adj[-1]
        rtn = close[-1] / close[0] - 1
        median = np.nanmedian(rtn)
        up_std = rtn[rtn > median].std()
        down_std = rtn[rtn < median].std()
        f = (up_std - down_std) / (up_std + down_std)
        return f
