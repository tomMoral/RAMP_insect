import numpy as np
from wavelet2D import WaveletTransform2D
from scipy import signal
import md5


class ScatteringTransform(object):
    """docstring for ScatteringTransform"""
    def __init__(self, mother_wavelet='daubechies4', level=1,
                 wlen=10):
        self.wt = WaveletTransform2D(mother_wavelet)
        self.lv = level
        t = np.linspace(-.5, .5, wlen + (1-(wlen % 2)))
        t = np.reshape(t, (-1, 1)) + np.reshape(t, (1, -1))

        self.phi = np.exp(t**2)
        self.phi /= (self.phi**2).sum()

        self.cache = {}

    def fit(self, f, layer=3):
        k = md5.new(f.flatten()).digest()
        if(k in self.cache.keys()):
            return self.cache[k]
        rpz = []
        Wphi = [[f]]
        for l in range(layer):
            rpz += [[]]
            Wphi += [[]]
            for i in range(len(Wphi[0])):
                sig = Wphi[0].pop(0)
                fh, wphi, l = self.wt.fit(abs(sig), self.lv)
                Wphi[-1] += wphi[0]
                if np.shape(fh) > self.phi.shape:
                    rpz[-1] += [abs(signal.convolve2d(fh, self.phi, mode='valid')).sum()]
                else:
                     rpz[-1] += [abs(fh).sum()]
            Wphi.remove([])
        self.cache[k] = rpz
        return rpz
