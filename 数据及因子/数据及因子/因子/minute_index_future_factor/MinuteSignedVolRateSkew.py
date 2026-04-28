from future_factor import FutureFactor
import numpy as np
from scipy import stats


class  MinuteSignedVolRateSkew(FutureFactor):
    '''
    Description: 
    Class: Volume_Stat
    Author: shentq modeified by liuz
    '''
    data_type='Future'
    instrument_type='main'
    days_past=6
    data_dict=dict()
    data_dict['Index_Id'] = {'000300.SH':['close','open']}
    data_dict['Future_Data'] = ['volume']

    normalize_size=20*237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):

        volume_array = data['volume'].values[:237*6].reshape(6, -1)
        vol_mean1 = np.nanmean(volume_array[:-1],axis=0)
        vol_mean2 = np.nanmean(volume_array[1:],axis=0)
        
        vol_ratio1 = volume_array[-1]/vol_mean1
        volume_today = data['volume'].values[237*6:]
        vol_ratio2 = volume_today/vol_mean2[:len(volume_today)]
        
        vol_ratio = np.append(vol_ratio1, vol_ratio2)

        index_close_list = data['close_000300.SH'].values
        index_open_list = data['open_000300.SH'].values
        rtn_list = [index_close_list[i] / index_open_list[i] - 1 for i in range(len(index_close_list))]
        
        
        signed_vol_ratio_list = []
        for i in range(30):
            vol_r = vol_ratio[-30:][i]
            rtn = rtn_list[-30:][i]
           
            signed_vol_ratio_list.append(vol_r*np.sign(rtn))
            
        factor = -stats.skew(signed_vol_ratio_list)
            
        return  factor
