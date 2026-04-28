from future_factor import FutureFactor
import numpy as np


class MinuteAllIndexTurnoverShrinkReturn(FutureFactor):
    '''
    Description: "mean(where((index_turnover_ratio_IC < shift(index_turnover_ratio_IC, 1)) & (index_turnover_ratio_IF < shift(index_turnover_ratio_IF, 1)
                    & (index_turnover_ratio_IH < shift(index_turnover_ratio_IH, 1), pct_chg(Index_ClosePx, 1), nan), 45),
                    index_turnover_ratio = Index_Turnover / (the average Index_Turnover at current time over past 5 trading days)"
    Class: PV_Corr
    Author: jinpx  modeified by liuz
    '''    
    data_type='Future'
    instrument_type='main'
    days_past=7
    data_dict=dict()
    data_dict['Index_Id'] = {'000300.SH':['close','amount'],'000905.SH':['amount'],'000016.SH':['amount']}

    normalize_size= 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):

        IF_close = data['close_000300.SH'].values
        IC_amt = data['amount_000905.SH'].values
        IF_amt = data['amount_000300.SH'].values
        IH_amt = data['amount_000016.SH'].values

        r_IF = np.diff(IF_close) / IF_close[:-1]
        IC_turnover_ratio = IC_amt[-237:] / np.nanmean(IC_amt[-1422:-237].reshape(5, 237), axis=0)
        IF_turnover_ratio = IF_amt[-237:] / np.nanmean(IF_amt[-1422:-237].reshape(5, 237), axis=0)
        IH_turnover_ratio = IH_amt[-237:] / np.nanmean(IH_amt[-1422:-237].reshape(5, 237), axis=0)

        N = 60
        f = np.nanmean(r_IF[-N:][np.logical_and.reduce([np.diff(IC_turnover_ratio)[-N:]<0, np.diff(IF_turnover_ratio)[-N:]<0, np.diff(IH_turnover_ratio)[-N:]<0])])
        return f
