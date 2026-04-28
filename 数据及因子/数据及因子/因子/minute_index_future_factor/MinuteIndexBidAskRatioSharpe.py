import numpy as np
from future_factor import FutureFactor

class MinuteIndexBidAskRatioSharpe(FutureFactor):
    '''
    Description: cs_mean(Shapre((TotalBidVol - TotalAskVol) / (TotalBidVol + TotalAskVol), 15)
    Class: Bid_Ask
    Author: liuz, modified by jinpx
    '''
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['TotalBidVol', 'TotalAskVol']
    normalize_size = 5 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        totalbidvol = data['TotalBidVol'].values
        totalaskvol = data['TotalAskVol'].values

        N = 15
        bid_ask_pressure = (totalbidvol[-N:] - totalaskvol[-N:]) / (totalbidvol[-N:] + totalaskvol[-N:])
        bid_ask_pressure_mean = np.nanmean(bid_ask_pressure, axis=0)
        bid_ask_pressure_std = np.nanstd(bid_ask_pressure, axis=0)
        f = np.nanmean(bid_ask_pressure_mean[bid_ask_pressure_std!=0] / bid_ask_pressure_std[bid_ask_pressure_std!=0])
        
        return f