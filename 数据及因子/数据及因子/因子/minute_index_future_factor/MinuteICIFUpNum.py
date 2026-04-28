from future_factor import FutureFactor
import numpy as np


class MinuteICIFUpNum(FutureFactor):
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = {}
    data_dict['Index_Id'] = {'000300.SH': ['close'], '000905.SH': ['close']}
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        lb = 237
        close_300 = data['close_000300.SH'].values
        close_500 = data['close_000905.SH'].values
        f = ((close_300[1:] > close_300[:-1])[-lb:] & (close_500[1:] > close_500[:-1])[-lb:]).sum()
        return f
