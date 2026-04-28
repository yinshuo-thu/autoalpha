from future_factor import FutureFactor
import numpy as np


class  MinuteNetAmtRatio(FutureFactor):
    '''
    Description: "(sum(where(pct_chg(ClosePx, 1) > rtn_mean + 2 * rtn_std, Turnover, 0), 120) - sum(where(pct_chg(ClosePx, 1) < rnt_mean - 2 * rtn_std, Turnover, 0), 120)) /
    (sum(where(pct_chg(ClosePx, 1) > rtn_mean + 2 * rtn_std, Turnover, 0), 120) + sum(where(pct_chg(ClosePx, 1) < rnt_mean - 2 * rtn_std, Turnover, 0), 120)),
    rtn_mean = mean(pct_chg(ClosePx, 1), 5days), rtn_std = std(pct_chg(ClosePx, 1), 5days) / (10 ** 0.5)"
    Class:MTM
    Author: shentq modeified by liuz
    '''
    data_type='Future'
    instrument_type='main'
    days_past=6
    data_dict=dict()
    data_dict['Future_Data'] = ['amount']
    data_dict['Continuous_Data'] = {'IF':['close']}

    normalize_size=237
    normalize_type = 'ts_rank'

    def calculate(self, data):

        close_lastdays = data['close_cont_IF'].values[:240*5]
        amount_lastdays = data['amount'].values[:240*5]

        rtn_list_lastdays = close_lastdays[1:]/close_lastdays[:-1]-1
        rtn_mean = np.nanmean(rtn_list_lastdays)
        rtn_std = np.nanstd(rtn_list_lastdays)/np.sqrt(10)

        close = data['close_cont_IF'].values
        amount = data['amount'].values
        rtn_list= close[1:]/close[:-1]-1

        turnover_up_array = amount[-240:][rtn_list[-240:] > rtn_mean+2*rtn_std]
        turnover_down_array =amount[-240:][rtn_list[-240:] < rtn_mean-2*rtn_std]

        if len(turnover_up_array) <= 0:
            turnover_up_sum = 0
        else:
            turnover_up_sum = np.nansum(turnover_up_array)

        if len(turnover_down_array) <= 0:
            turnover_down_sum = 0
        else:
            turnover_down_sum = np.nansum(turnover_down_array)

        if turnover_down_sum == 0 and turnover_up_sum == 0:
            factor = 0
        else:
            factor = (turnover_up_sum - turnover_down_sum) / (turnover_up_sum + turnover_down_sum)

        return  factor
