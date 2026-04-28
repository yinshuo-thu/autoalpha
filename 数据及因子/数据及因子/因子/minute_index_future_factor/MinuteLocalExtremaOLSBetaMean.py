from future_factor import FutureFactor
import numpy as np


class MinuteLocalExtremaOLSBetaMean(FutureFactor):
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 5
    data_dict = {}
    data_dict['Index_Id'] = {'000300.SH': ['close']}
    normalize_size = 60
    normalize_type = 'ts_rank'
    
    def get_local_extrema(self, price):
        threshold = 2 * np.nanstd(price[1:] / price[:-1] - 1)
        localhighs = []
        locallows = []
        prel = 0
        preh = 0
        lh = 0
        ll = 0
        for i in range(len(price)):
            if price[lh] >= price[prel] * (1 + threshold) and price[lh] >= price[i] * (1 + threshold):
                localhighs.append(lh)
                lh = i
                prel = i
            else:
                if price[i] < price[lh] / (1 + threshold) or price[i] > price[lh]:
                    lh = i
                if price[i] <= price[prel]:
                    prel = i
            if price[ll] <= price[preh] / (1 + threshold) and price[ll] <= price[i] / (1 + threshold):
                locallows.append(ll)
                ll = i
                preh = i
            else:
                if price[i] > price[ll] * (1 + threshold) or price[i] < price[ll]:
                    ll = i
                if price[i] >= price[preh]:
                    preh = i
        if price[lh] >= price[prel] * (1 + threshold):
            localhighs.append(lh)
        if price[ll] <= price[preh] / (1 + threshold):
            locallows.append(ll)
        return localhighs, locallows
    
    def calculate(self, data):
        lb = 5 * 237
        close = data['close_000300.SH'].values[-lb:]
        lh, ll = self.get_local_extrema(close)
        if (len(ll) >= 3) & (len(lh) >= 3):
            beta_h = np.nanmean((close[lh] - np.nanmean(close[lh])) * (lh - np.nanmean(lh))) / np.nanvar(lh)
            beta_l = np.nanmean((close[ll] - np.nanmean(close[ll])) * (ll - np.nanmean(ll))) / np.nanvar(ll)
            f = (beta_l + beta_h) / 2
        else:
            f = 0
        return f
