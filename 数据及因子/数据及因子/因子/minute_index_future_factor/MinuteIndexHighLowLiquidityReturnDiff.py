import numpy as np
from future_factor import FutureFactor

class MinuteIndexHighLowLiquidityReturnDiff(FutureFactor):
    '''
    Description: high_low_diff((AskP0-BidP0)/volume_ratio, 20), cs_mean(r)
    Class: Group_Stat
    Author: jinpx, modified by jinpx
    '''
    data_type = 'IndexStock'
    instrument_type = 'main'
    days_past = 6
    data_dict = dict()
    data_dict['Stock'] = ['close', 'adjfactor', 'volume', 'AskP0', 'BidP0']
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        
        close = data['close'].values
        adjfactor = data['adjfactor'].values
        volume = data['volume'].values
        ask_0_price = data['AskP0'].values
        bid_0_price = data['BidP0'].values
        
        close_adj = close * adjfactor
        r = np.diff(close_adj[-238:], n=1, axis=0) / close_adj[-238:][:-1]
        spread = (ask_0_price[-238:] - bid_0_price[-238:]) / (ask_0_price[-238:] + bid_0_price[-238:])
        volume_ratio = volume[-237:] / np.nanmean(volume[-1422:-237].reshape(5, 237, len(close[0])), axis=0)
        
        N = 120
        liquidity = volume_ratio[-N:] / spread[-N:]
        liquidity[np.isinf(liquidity)] = np.nan
        liquidity_mean = np.nanmean(liquidity, axis=0)
        liquidity_high_limit = np.percentile(liquidity_mean[~np.isnan(liquidity_mean)], 80)
        liquidity_low_limit = np.percentile(liquidity_mean[~np.isnan(liquidity_mean)], 20)
        N = 20
        r_mean = np.nanmean(r[-N:], axis=0)
        f = np.nanmean(r_mean[liquidity_mean>liquidity_high_limit]) - np.nanmean(r_mean[liquidity_mean<liquidity_low_limit])
            
        return f