import numpy as np
from future_factor import FutureFactor

class MinuteIndexOmega(FutureFactor):
    '''
    Description: cs_mean(ts_omega(close, 150))
    Class: Return_Risk
    Author: jinpx, modified by jinpx
    '''
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['close', 'adjfactor']
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        close = data['close'].values
        adjfactor = data['adjfactor'].values
        close_adj = close * adjfactor
        
        N = 150
        r = (np.diff(close_adj, axis=0) / close_adj[:-1])[-N:] 
        
        omega = []
        for i in range(len(r[0])):
            r_sorted = np.sort(r[:,i])
            cdf = np.arange(0, 1, 1/len(r_sorted)) + 1/len(r_sorted)
            p = 1 / len(r_sorted)
            positive = np.sum((1-cdf[r_sorted>0])*p)
            negative = np.sum(cdf[r_sorted<0]*p)
            omega_one = positive / negative
            if omega_one == np.inf:
                omega_one = np.nan
            omega = np.append(omega, omega_one)
        f = np.nanmean(omega)
                        
        return f