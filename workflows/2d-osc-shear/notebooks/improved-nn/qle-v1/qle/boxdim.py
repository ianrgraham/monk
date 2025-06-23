from numba.experimental import jitclass
import numpy as np
import numba

from .data import numba_scalar, np_scalar


spec = [
    ('l', numba_scalar[:]),
    ('l_inv', numba_scalar[:]),
    ('hi', numba_scalar[:]),
    ('lo', numba_scalar[:]),
    ('xy', numba_scalar),
    ('xz', numba_scalar),
    ('yz', numba_scalar),
    ('periodic', numba.boolean[:])
]
@jitclass(spec)
class BoxDim:

    def __init__(self, box, is_periodic=None):
        if is_periodic is None:
            is_periodic = np.array([True, True, True])

        assert len(box) == 6
        assert len(is_periodic) == 3
        self.l = box[:3]
        self.l_inv = numba_scalar(1.0) / self.l
        self.hi = self.l / 2
        self.lo = -self.hi
        self.xy = box[3]
        self.xz = box[4]
        self.yz = box[5]
        self.periodic = is_periodic

    
    def wrap_smol(self, w):
        if self.periodic[2]:
            if w[2] >= self.hi[2]:
                w[2] -= self.l[2]
                w[1] -= self.l[2] * self.yz
                w[0] -= self.l[2] * self.xz
            elif w[2] < self.lo[2]:
                w[2] += self.l[2]
                w[1] += self.l[2] * self.yz
                w[0] += self.l[2] * self.xz
            
        if self.periodic[1]:
            if w[1] >= self.hi[1]:
                w[1] -= self.l[1]
                w[0] -= self.l[1] * self.xy
            elif w[1] < self.lo[1]:
                w[1] += self.l[1]
                w[0] += self.l[1] * self.xy

        if self.periodic[0]:
            if w[0] >= self.hi[0]:
                w[0] -= self.l[0]
            elif w[0] < self.lo[0]:
                w[0] += self.l[0]

        return w
    
    def wrap(self, ws):
        wout = np.zeros_like(ws)
        for i in range(len(ws)):
            wout[i] = self.wrap_smol(ws[i])

        return wout
