from future_factor import FutureFactor
import numpy as np


class  MinuteIndexBSUniqueDiff30Ma(FutureFactor):
    '''
    Description: ts_mean(cs_sum(SellUniqueOrderNum) / cs_sum(SellTradeNum) - cs_sum(BuyUniqueOrderNum) / cs_sum(BuyTradeNum), 30)
    Class:Buy_Sell
    Author: shentq  modeified by liuz
    '''
    data_type = 'IndexStock'
    instrument_type='main'
    days_past=1
    data_dict=dict()
    data_dict['Stock'] = ['BuyUniqueOrderNum', 'SellUniqueOrderNum', 'BuyTradeNum', 'SellTradeNum']

    normalize_size=20*237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):

        BuyUniqueOrderNum = data['BuyUniqueOrderNum'].values[-30:]
        SellUniqueOrderNum = data['SellUniqueOrderNum'].values[-30:]
        BuyTradeNum = data['BuyTradeNum'].values[-30:]
        SellTradeNum = data['SellTradeNum'].values[-30:]
        
        buy_unique_num_list =np.nansum(BuyUniqueOrderNum,axis=1).tolist()
        sell_unique_num_list = np.nansum(SellUniqueOrderNum,axis=1).tolist()
        buy_num_list = np.nansum(BuyTradeNum,axis=1).tolist()
        sell_num_list = np.nansum(SellTradeNum,axis=1).tolist()

        buy_unique_ratio_list = []
        sell_unique_ratio_list = []

        buy_sell_unique_diff_list = []
        
        for i in range(len(buy_unique_num_list)):
            if buy_num_list[i] == 0:
                buy_unique_ratio_list.append(0)
            else:
                buy_unique_ratio_list.append(buy_unique_num_list[i] / buy_num_list[i])

            if sell_num_list[i] == 0:
                sell_unique_ratio_list.append(0)
            else:
                sell_unique_ratio_list.append(sell_unique_num_list[i] / sell_num_list[i])

            buy_sell_unique_diff_list.append(sell_unique_ratio_list[-1] - buy_unique_ratio_list[-1])
        
        factor = np.nanmean(buy_sell_unique_diff_list[-30:])
        return factor
