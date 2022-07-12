"""define misc. force compute classes in python"""

import hoomd
import numpy as np
import gsd.hoomd

METHOD = "SLOW"


class FrictionForce(hoomd.md.force.Custom):

    def __init__(self, nlist: hoomd.md.nlist.NeighborList):
        super().__init__()
        self._nlist = nlist

    def set_forces(self):

        # match METHOD:
        #     case "SLOW":
        snapshot: gsd.hoomd.Snapshot = self._state.get_snapshot()
        positions: np.ndarray = snapshot.particles.position

        self._state.set_snapshot(snapshot)

        # with self.cpu_local_force_arrays as arrays:
        #     arrays.force[:] = -5
        #     arrays.torque[:] = 3
        #     arrays.potential_energy[:] = 27
        #     arrays.virial[:] = np.arange(6)[None, :]
