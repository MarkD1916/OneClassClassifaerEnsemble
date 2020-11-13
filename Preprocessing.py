import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
class Features(BaseEstimator, TransformerMixin):
    def __init__(self, function, band):
        self.function = function
        self.band = band

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):

        def corr(data):  # подготавливает выборку признаков crosscorr
            coef = [
                np.triu(np.asmatrix((np.corrcoef(data[i]))), 1)[np.triu(np.asmatrix((np.corrcoef(data[i]))), 1) != 0]
                for i in range(len(data))]
            coef = np.array(coef)
            return coef

        def var(data):
            ch = data.shape[1]
            data = np.array([[np.var(data[event][channel]) for channel in range(ch)] for event in range(data.shape[0])])
            return data

        def fft(data, freq=1000, band=False, stop=136, step=2, start=70, fit=5):  # подготавливает выборку признаков fft
            if band == False:
                freq_band = [[i, i + fit] for i in range(start, stop, step)]
            else:
                freq_band = sorted(band, key=lambda x: x[0])
            def fft(data, freq, dia_from, dia_to):
                sfreq = freq
                ch = data.shape[1]
                N = data.shape[2]
                spectrum = np.abs((np.fft.rfft(data)))
                freqs = np.fft.rfftfreq(N, 1. / sfreq)
                mask_signal = np.all([[(freqs >= dia_from)], [(freqs <= dia_to)]], axis=0)[0]
                data = np.array(
                    [[np.mean(spectrum[event][channel][mask_signal]) for channel in range(ch)] for event in
                     range(data.shape[0])])
                return data
            start = fft(data, freq, freq_band[0][0], freq_band[0][1])
            if len(freq_band) > 1:
                for i in range(1, len(freq_band)):
                    end = fft(data, freq, freq_band[i][0], freq_band[i][1])
                    array_stack = np.hstack((start, end))
                    start = array_stack
                return array_stack
            else:
                return start

        if self.function == 'cor':
            X = corr(X)
            return X
        if self.function == 'fft':
            X = fft(X, band=self.band)
            return X

