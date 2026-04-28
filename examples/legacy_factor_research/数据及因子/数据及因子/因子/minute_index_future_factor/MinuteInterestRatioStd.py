import numpy as np
from future_factor import FutureFactor

class MinuteInterestRatioStd(FutureFactor):
    '''
    Description: 
    Class: 
    Author: jinpx
    '''    
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Future_Data'] = ['interest']
    normalize_size = 5 * 240
    normalize_type = 'ts_rank'

    def calculate(self, data):
        
        interest = data['interest'].values
        
        N = 20
        interest_ratio = interest[-N:] / np.nansum(interest[-N:]) 

        f = - np.nanstd(interest_ratio[-N:])
        
        return f