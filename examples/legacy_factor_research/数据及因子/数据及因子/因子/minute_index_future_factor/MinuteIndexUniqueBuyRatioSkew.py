from future_factor import FutureFactor
import numpy as np
from scipy import stats

class  MinuteIndexUniqueBuyRatioSkew(FutureFactor):
    '''
    Description: ts_mean(cs_skew(BuyUniqueOrderNum / BuyTradeNum), 5)
    Class:Buy_Sell
    Author: shentq  modeified by liuz
    '''
    data_type = 'IndexStock'
    instrument_type='main'
    days_past=1
    data_dict=dict()
    data_dict['Stock'] = ['BuyUniqueOrderNum', 'BuyTradeNum']

    
    normalize_size= 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):

        BuyTradeNum = data['BuyTradeNum'].values
        BuyUniqueOrderNum = data['BuyUniqueOrderNum'].values
        df_buy_unique_ratio = BuyUniqueOrderNum / BuyTradeNum
        s = stats.skew(df_buy_unique_ratio[-30:],axis=1,nan_policy='omit',bias=False)

        factor = np.nanmean(np.array(s))

        return factor
