import freud
from hoomd.md.force import Force, Custom

import numpy as np
from numba import njit


@njit
def _diff_with_rtag(bias_pos, pos, rtags):
    """Compute the difference between bias pos and current pos.
    The rtags are required to convert indices of the current position
    back to their "tags" (i.e. the indices of the bias positions), which
    should be the indices for the bias pos. The returned difference
    reflects the same indices of "pos"
    """
    out = np.zeros_like(pos)
    n = len(rtags)
    for tag_idx in range(n):
        idx = rtags[tag_idx]
        out[idx] = pos[idx] - bias_pos[tag_idx]
    return out


@njit
def _mimse_force_energy(diff_pos, force, energy, bias_U_0, bias_U_sigma,
                        bias_exclude):
    """Compute the MIMSE force and energy."""
    N = len(diff_pos)
    mimse_forces = np.zeros_like(diff_pos)
    bias_factor = 1.0 - np.sum(np.square(diff_pos)) / (np.square(bias_U_sigma))
    for i in range(N):
        energy[i] += bias_U_0 * bias_factor**2 / N * (bias_factor > 0)
        mimse_forces[
            i] = 4 * bias_U_0 / bias_U_sigma**2 * bias_factor * diff_pos[i] * (
                bias_factor > 0)

    mimse_forces -= np.sum(mimse_forces, axis=0) / N

    for i in range(N):
        force[i] += mimse_forces[i]


class MIMSEForce(Custom):

    def __init__(self, box, delta=0.1):
        super().__init__()

        self.delta = delta
        self.bias_scales = []
        self.bias_rads = []
        self.bias_positions = []
        self.bias_exclude = []

        self.box = freud.box.Box.from_box(box)
        self.last_nn_pos = None
        self.nn = []

        self.remove_cand = set()

    def add_bias(self, bias_scale, bias_rad, bias_position, bias_exclude):
        self.bias_scales.append(bias_scale)
        self.bias_rads.append(bias_rad)
        self.bias_positions.append(bias_position)
        self.bias_exclude.append(bias_exclude)
        self.nn.append(len(self.bias_positions) - 1)
        self.random_kick(bias_rad)

    def remove_bias(self, index):
        self.bias_scales.pop(index)
        self.bias_rads.pop(index)
        self.bias_positions.pop(index)
        self.bias_exclude.pop(index)

    def random_kick(self, bias_rad):
        """Randomly kick the bias positions.
        This is used to kick the force and energy.
        """
        snap = self._state.get_snapshot()
        pos = snap.particles.position

        kick = np.random.normal(0, 1.0, size=pos.shape)
        if self._state.box.is2D:
            kick[:, 2] = 0
        kick /= np.linalg.norm(kick)
        kick *= 0.05 * bias_rad
        pos += kick

        pos = self.box.wrap(pos)

        snap.particles.position[:] = pos
        self._state.set_snapshot(snap)

    def find_nn(self, pos, rtags):
        if self.last_nn_pos is None:
            nn_euclidean_distance = 2 * self.delta**2
        else:
            diff = self.box.wrap(_diff_with_rtag(self.last_nn_pos, pos, rtags))
            nn_euclidean_distance = np.sum(np.square(diff))
        if nn_euclidean_distance > self.delta**2:
            self.nn.clear()
            pos = self._state.get_snapshot().particles.position
            self.last_nn_pos = pos

            for i in range(len(self.bias_positions)):
                bias_pos = self.bias_positions[i]
                bias_rad = self.bias_rads[i]

                diff = np.sum(np.square(self.box.wrap(pos - bias_pos)))
                if diff <= (bias_rad * (1 + self.delta))**2:
                    self.nn.append(i)

    def set_forces(self, timestep):

        with self.cpu_local_force_arrays as arrays:
            with self._state.cpu_local_snapshot as data:

                pos = data.particles.position._coerce_to_ndarray()
                rtags = data.particles.rtag._coerce_to_ndarray()

                self.find_nn(pos, rtags)

                for i in self.nn:

                    bias_pos = self.bias_positions[i]

                    diff = _diff_with_rtag(bias_pos, pos, rtags)
                    diff = self.box.wrap(diff)

                    force = arrays.force._coerce_to_ndarray()
                    energy = arrays.potential_energy._coerce_to_ndarray()

                    bias_U_0 = self.bias_scales[i]
                    bias_U_sigma = self.bias_rads[i]
                    bias_exclude = self.bias_exclude[i]

                    _mimse_force_energy(diff, force, energy, bias_U_0,
                                        bias_U_sigma, bias_exclude)
