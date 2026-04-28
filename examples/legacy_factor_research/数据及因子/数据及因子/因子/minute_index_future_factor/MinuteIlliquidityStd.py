import numpy as np
from future_factor import FutureFactor

class MinuteIlliquidityStd(FutureFactor):
    '''
    Description: - Std(AbsDistance/amount, 15)
    Class: 
    Author: jinpx
    '''    
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 6
    data_dict = dict()
    data_dict['Continuous_Data'] = {'IF': ['AbsDistance', 'amount']}
    normalize_size = 5 * 237
    normalize_type = 'ts_rank'

    def calculate(self, data):

        AbsDistance = data['AbsDistance_cont_IF'].values
        amount = data['amount_cont_IF'].values
        
        AbsDistance_ratio = AbsDistance[-240:] / np.nanmean(AbsDistance[-1440:-240].reshape(5, 240), axis=0)
        amount_ratio = amount[-240:] / np.nanmean(amount[-1440:-240].reshape(5, 240), axis=0)
        
        illiquidity = AbsDistance_ratio / amount_ratio
        
        N = 15
        f = - np.nanstd(illiquidity[-N:])
        
        return f