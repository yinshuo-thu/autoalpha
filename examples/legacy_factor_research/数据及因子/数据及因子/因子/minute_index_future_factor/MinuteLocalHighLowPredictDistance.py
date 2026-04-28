import numpy as np
from future_factor import FutureFactor

class MinuteLocalHighLowPredictDistance(FutureFactor):
    '''
    Description: 
    Class: Local_High_Low
    Author: liuz, modified by jinpx
    '''    
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 10
    data_dict = dict()
    data_dict['Index_Id'] = {'000300.SH':['close']}
    normalize_size = 5 * 237
    normalize_type = 'ts_rank'
    
    def getlocallows(self, price, threshold=0.01):
        prehigh = []
        posthigh = []
        locallow = []
        preh = 0 
        posth = 0
        ll = 0
        for i in range(len(price)):
            if price[ll] <= price[preh]/(1+threshold):
                if price[ll] <= price[posth]/(1+threshold):
                    prehigh.append(preh)
                    locallow.append(ll)
                    ll = i-1
                    preh = i-1
                    posth = i-1
                    if price[i] >= price[preh]:
                        preh = i
                        ll = i
                        posth = i
                    else:
                        ll = i
                        posth = i

                else:
                    if price[i] <= price[ll]:
                        ll = i
                        posth = i
                    else:
                        posth = i
            else:
                if price[i] >= price[preh]:
                    preh = i
                    ll = i
                    posth = i
                else:
                    ll = i
                    posth = i
        return locallow

    def getlocalhighs(self,  price, threshold=0.01):

        prelow = []
        postlow = []
        localhigh = []
        prel = 0 
        postl = 0
        lh = 0
        for i in range(len(price)):
            if price[lh] >= price[prel]*(1+threshold):
                if price[lh] >= price[postl]*(1+threshold):
                    prelow.append(prel)
                    localhigh.append(lh)
                    lh = i-1
                    prel = i-1
                    postl = i-1
                    if price[i] <= price[prel]:
                        prel = i
                        lh = i
                        postl = i
                    else:
                        lh = i
                        postl = i
                else:
                    if price[i] >= price[lh]:
                        lh = i
                        postl = i
                    else:
                        postl = i
            else:
                if price[i] <= price[prel]:
                    prel = i
                    lh = i
                    postl = i
                else:
                    lh = i
                    postl = i
        return localhigh
    
    def calculate(self, data):
                
        index_close = data['close_000300.SH'].values
        
        lows = self.getlocallows(index_close, threshold=0.005)
        highs = self.getlocalhighs(index_close, threshold=0.005)
        
        if len(lows) > len(highs):
            lows = lows[1:]
        elif len(lows) < len(highs):
            highs = highs[1:]

        if len(lows) == 0 or len(highs) == 0:
            f = 0
        elif lows[0] < highs[0]:
            swing = np.nanmean(index_close[highs] / index_close[lows] - 1)
            re = index_close[-1] / index_close[highs[-1]] - 1
            f = np.abs(re - swing)
        else:
            swing = np.nanmean(index_close[lows] / index_close[highs] - 1)
            re = index_close[-1] / index_close[lows[-1]] - 1
            f = np.abs(swing - re)
        
        return f