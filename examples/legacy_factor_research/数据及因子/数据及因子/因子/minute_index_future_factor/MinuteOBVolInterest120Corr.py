from future_factor import FutureFactor
import numpy as np


class  MinuteOBVolInterest120Corr(FutureFactor):
    '''
    Description: corr(Interest, AskVol + BidVol, 120)
    Class: Bid_Ask
    Author:  shentq modeified by liuz
    '''
    data_type='Future'
    instrument_type='main'
    days_past=1
    data_dict=dict()
    data_dict['Future_Data'] = ['AskVol', 'BidVol','interest']

    normalize_size=5*240
    normalize_type = 'ts_rank'
    
    def calculate(self, data):

        AskVol = data['AskVol'].values 
        BidVol = data['BidVol'].values 
        Interest = data['interest'].values 

        ob_vol = AskVol+BidVol
        factor = np.corrcoef(ob_vol[-240:], Interest[-240:])[0,1]
        return  factor
