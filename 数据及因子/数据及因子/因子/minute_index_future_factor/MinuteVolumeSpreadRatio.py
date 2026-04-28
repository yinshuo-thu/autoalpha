import numpy as np
from future_factor import FutureFactor

class MinuteVolumeSpreadRatio(FutureFactor):
    '''
    Description: mean(volume_ratio / ((AskP0 - BidP0) / (AskP0 + BidP0)), 45)
                 volume_ratio = volume / (the average volume at current time over past 5 trading days)
    Class: Liquidity
    Author: jinpx, modified by jinpx
    '''    
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 6
    data_dict = dict()
    data_dict['Continuous_Data'] = {'IF':['volume', 'AskP0', 'BidP0']}
    normalize_size = 5 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        volume = data['volume_cont_IF'].values
        ask_0 = data['AskP0_cont_IF'].values
        bid_0 = data['BidP0_cont_IF'].values
        
        spread = (ask_0 - bid_0) / (ask_0 + bid_0)
        volume_ratio = volume[-240:] / np.nanmean(volume[-1440:-240].reshape(5, 240), axis=0)
            
        N = 45
        liquidity = volume_ratio[-N:] / spread[-N:] 
        liquidity[np.isinf(liquidity)] = np.nan

        f = np.nanmean(liquidity)
        
        return f