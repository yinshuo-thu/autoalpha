from future_factor import FutureFactor
import numpy as np


class MinuteDistanceAmountOLSBeta(FutureFactor):
    data_type = 'Future'
    instrument_type = 'main'
    days_past = 1
    data_dict = {}
    data_dict['Index_Id'] = {'000300.SH': ['open', 'close', 'high', 'low', 'amount']}
    normalize_size = 20 * 237
    normalize_type = 'ts_rank'
    
    def calculate(self, data):
        lb = 20
        open_px = data['open_000300.SH'].values
        close = data['close_000300.SH'].values
        high = data['high_000300.SH'].values
        low = data['low_000300.SH'].values
        amount = data['amount_000300.SH'].values
        x = amount
        y = 2 * (high - low) - (np.maximum(open_px, close) - np.minimum(open_px, close))
        is_ava = ~((x == 0) | (y == 0) | np.isnan(x) | np.isnan(y))
        x = x[is_ava][-lb:]
        y = y[is_ava][-lb:]
        f = -1 / (x.dot(x)) * (x.dot(y))
        return f
