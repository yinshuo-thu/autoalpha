import numpy as np
import pandas as pd
from future_factor import FutureFactor

class MinuteIndexAsk1Bid1Beta(FutureFactor):

    data_type = 'IndexStock'
    days_past = 1
    data_dict = dict()
    data_dict['Stock'] = ['Bid1AmtMean', 'Ask1AmtMean','weight' ]
    normalize_size =  120
    normalize_type = 'ts_rank'

    def calculate(self, data):

        lb = 237


        Ask1AmtMean =  data['Ask1AmtMean'].values[-lb:]
        Bid1AmtMean =  data['Bid1AmtMean'].values[-lb:]
        weight = data['weight'].values[-1]


        BETA= []
        for i in range(len(Ask1AmtMean[0])):
            Ask1AmtMean_s = Ask1AmtMean[:,i] 
            Bid1AmtMean_s = Bid1AmtMean[:,i]
            beta = (np.nansum(Ask1AmtMean_s*Bid1AmtMean_s)*len(Ask1AmtMean_s)-np.nansum(Ask1AmtMean_s)*np.nansum(Bid1AmtMean_s))/(len(Bid1AmtMean_s)*np.nansum(Bid1AmtMean_s*Bid1AmtMean_s)-np.nansum(Bid1AmtMean_s)*np.nansum(Bid1AmtMean_s))
            BETA.append(beta)
        factor = np.nanmean(BETA*weight)

        return factor