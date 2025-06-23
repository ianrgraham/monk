from numba.experimental import jitclass
import numpy as np

from .data import numba_scalar, np_scalar

spec = [
    ('epsilon', numba_scalar[:]),
    ('sigma', numba_scalar[:]),
    ('cutoff', numba_scalar[:]),
]
@jitclass(spec)
class KATypeParamDict:
    def __init__(self):
        self.epsilon = np.array([1.0, 1.5, 0.5], dtype=np_scalar)
        self.sigma = np.array([1.0, 0.8, 0.88], dtype=np_scalar)
        self.cutoff = np.array([2.5, 2.5*0.8, 2.5*0.88], dtype=np_scalar)

    @property
    def ntypes(self):
        return 2
    
    @staticmethod
    def max_cutoff():
        return 2.5
    
    def get_params(self, i, j):
        return self.epsilon[i+j], self.sigma[i+j], self.cutoff[i+j]

    @staticmethod
    def _grad(r, epsilon, sigma, cutoff):
        if r > cutoff:
            return 0.0
        x = sigma / r
        x2 = x * x
        r2 = r * r
        x4 = x2 * x2
        x6 = x4 * x2
        return - 12.0 * epsilon * (2.0 * x6 * x6 - x6) / r2

    @staticmethod
    def _grad2(r, epsilon, sigma, cutoff):
        if r > cutoff:
            return 0.0
        x = sigma / r
        r2 = r * r
        r4 = r2 * r2
        x2 = x * x
        x4 = x2 * x2
        x6 = x4 * x2
        # comute the second derivative of the pair potential
        return epsilon * (168.0 * x6 * x6 - 48.0 * x6) / r4

    @staticmethod
    def _grad3(r, epsilon, sigma, cutoff):
        if r > cutoff:
            return 0.0
        x = sigma / r
        r2 = r * r
        r4 = r2 * r2
        r6 = r4 * r2
        x2 = x * x
        x4 = x2 * x2
        x6 = x4 * x2
        # comute the second derivative of the pair potential
        return - epsilon * (1344.0 * x6 * x6 - 240 * x6) / r6
