import numpy as np
import matplotlib.pyplot as plt
from math import log


class WaveletTransform2D(object):
    '''Class handeling Wavelet Transform

    Parameters
    ----------
    mother_wavelet: str or n-array, optional (default: 'daubechies4')
        Mother wavelet function. One can use an implemented version
        in {'daubechies4'}, or pass an array containing the h-filter
        coefficient.
    '''
    def __init__(self, mother_wavelet='daubechies4'):
        # Low pass filter
        if mother_wavelet == 'daubechies4':
            h = np.array([0, .482962913145, .836516303738,
                          .224143868042, -.129409522551])
        self.h = h

        # High pass filter
        u = np.array([(-1)**k for k in range(1, len(h))])
        self.g = np.r_[0, h[-1:0:-1]*u]

    def _vtransform(self, f):
        '''Compute the wavelet transform for signal f,
        return a t-uple (f*phi, f*psi)
        '''
        fht = [np.convolve(self.h, fh, mode='same')[::2]
               for fh in f]
        fgt = [np.convolve(self.h, fh, mode='same')[::2]
               for fh in f]
        return fht, fgt

    def _htransform(self, f):
        '''Compute the wavelet transform for signal f,
        return a t-uple (f*phi, f*psi)
        '''
        fht = np.transpose([np.convolve(self.h, fv, mode='same')[::2]
               for fv in np.transpose(f)])
        fgt = np.transpose([np.convolve(self.h, fv, mode='same')[::2]
               for fv in np.transpose(f)])
        return fht, fgt


    def fit(self, f, level=None):
        '''Compute the WaveletTransform until a level
        return a t-uple (fc, Wphi, lc)

        Parameters
        ----------
        level: int, optional (default: None)
            maximal level for the Wavelet Transform
            If not specified,

        Return
        ------
        fc: low pass component of the WT
        Wphi: list of the successive high frequency component
        lc: list of the successive size
        '''
        Wphi = []
        lf = []
        aa = f
        N = len(f)
        if level is None:
            level = int(log(N,2))-1
        level = min(level, int(log(N,2))-1)
        for l in range(level):
            aa, ah, av, dd = self.transform(aa)
            Wphi.append((ah, av, dd))
            lf.append(len(ah))
        return aa, Wphi, lf

    def transform(self, f):
        a, d = self._vtransform(f)
        aa, ah = self._htransform(a)
        av, dd = self._htransform(d)
        return aa, ah, av, dd

    def _upsample(self, f):
        uf = np.zeros(2*len(f))
        uf[::2] = f
        return uf
