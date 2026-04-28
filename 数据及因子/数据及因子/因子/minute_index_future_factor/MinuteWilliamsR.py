from future_factor import FutureFactor
import numpy as np


class MinuteWilliamsR(FutureFactor):
    '''
    Description: -(TodayHigh - close) / (TodayHigh - TodayLow)
    Class: Return_Risk
    Author: jinpx, modified by hefj
    '''
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 0
    data_dict = {}
    data_dict['Future_Data'] = ['close', 'TodayHigh', 'TodayLow']
    normalize_size = 1 * 240
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        f = -(data['TodayHigh'].values[-1] - data['close'].values[-1]) / (data['TodayHigh'].values[-1] - data['TodayLow'].values[-1])
        return f
