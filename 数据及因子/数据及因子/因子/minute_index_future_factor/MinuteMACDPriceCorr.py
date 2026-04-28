from future_factor import FutureFactor
import numpy as np


class  MinuteMACDPriceCorr(FutureFactor):
    '''

'''
    data_type='Future'
    instrument_type='main'
    days_past= 3
    data_dict=dict()
    data_dict['Index_Id'] = {'000300.SH':['close']}

    
    normalize_size= 120
    normalize_type = 'ts_rank'
    
    def calculate(self, data):

        
        def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
            """
            Calculates the exponential moving average over a vector.
            Will fail for large inputs.
            :param data: Input data
            :param alpha: scalar float in range (0,1)
                The alpha parameter for the moving average.
            :param offset: optional
                The offset for the moving average, scalar. Defaults to data[0].
            :param dtype: optional
                Data type used for calculations. Defaults to float64 unless
                data.dtype is float32, then it will use float32.
            :param order: {'C', 'F', 'A'}, optional
                Order to use when flattening the data. Defaults to 'C'.
            :param out: ndarray, or None, optional
                A location into which the result is stored. If provided, it must have
                the same shape as the input. If not provided or `None`,
                a freshly-allocated array is returned.
            """
            data = np.array(data, copy=False)

            if dtype is None:
                if data.dtype == np.float32:
                    dtype = np.float32
                else:
                    dtype = np.float64
            else:
                dtype = np.dtype(dtype)

            if data.ndim > 1:
                # flatten input
                data = data.reshape(-1, order)

            if out is None:
                out = np.empty_like(data, dtype=dtype)
            else:
                assert out.shape == data.shape
                assert out.dtype == dtype

            if data.size < 1:
                # empty input, return empty array
                return out

            if offset is None:
                offset = data[0]

            alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

            # scaling_factors -> 0 as len(data) gets large
            # this leads to divide-by-zeros below
            scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                                       dtype=dtype)
            # create cumulative sum array
            np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                        dtype=dtype, out=out)
            np.cumsum(out, dtype=dtype, out=out)

            # cumsums / scaling
            out /= scaling_factors[-2::-1]

            if offset != 0:
                offset = np.array(offset, copy=False).astype(dtype, copy=False)
                # add offsets
                out += offset * scaling_factors[1:]

            return out
        
        def MACD(close):

            window_short = 12
            window_long = 26
            window_mid = 9

            ema_short = ewma_vectorized(close, 2/(window_short+1))
            ema_long = ewma_vectorized(close, 2/(window_long+1))
            dif = ema_short[-(len(close)-window_long):] - ema_long[-(len(close)-window_long):]
            dea = ewma_vectorized(dif, 2/(window_mid+1))

            macd = dif[-(len(dif)-window_mid):] - dea[-(len(dif)-window_mid):]

            return macd
        
        lb = 237*2
        index_close = data['close_000300.SH'].values

        macdhist = MACD(index_close)

        
        factor = np.corrcoef(macdhist[-lb:], index_close[-lb:])[0][1]
                
        return factor