import numpy as np
from future_factor import FutureFactor


class MinuteAccUpDownAmountRatio(FutureFactor):
    
    def __init__(self):
        super().__init__()
        self.acc_up_amount = 0
        self.acc_down_amount = 0
        self.acc_up_amount_list = []
        self.acc_down_amount_list = []

    data_type = 'Future'
    days_past = 1
    data_dict = dict()
    data_dict['Index_Id'] = {'000300.SH': ['close', 'amount']}
    normalize_size = 1 * 237
    normalize_type = 'ts_rank'

    def calculate(self, data):

        close = data['close_000300.SH'].values
        amount = data['amount_000300.SH'].values

        if close[-1] < close[-2]:
            self.acc_up_amount = amount[-1]
            self.acc_down_amount += amount[-1]
        else:
            self.acc_up_amount += amount[-1]
            self.acc_down_amount = amount[-1]

        self.acc_up_amount_list.append(self.acc_up_amount)
        self.acc_down_amount_list.append(self.acc_down_amount)

        factor_value = np.nanmean(np.array(self.acc_up_amount_list[-60:]) / np.array(self.acc_down_amount_list[-60:]))

        if np.isinf(factor_value):
            factor_value = 1

        return factor_value