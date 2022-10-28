import hoomd
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyFrom, OnlyTypes
from hoomd.logging import log
from hoomd.mesh import Mesh
from hoomd.md import _md
from hoomd._nlist_plugin import _nlist_plugin
from hoomd.operation import _HOOMDBaseObject

class NEBCell(hoomd.md.nlist.NeighborList):

    def __init__(self,
                 buffer,
                 segments,
                 exclusions=('bond',),
                 rebuild_check_delay=1,
                 check_dist=True,
                 deterministic=False,
                 mesh=None):

        self.segments = segments

        super().__init__(buffer, exclusions, rebuild_check_delay, check_dist,
                         mesh)

        self._param_dict.update(
            ParameterDict(deterministic=bool(deterministic)))

    def _attach(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            nlist_cls = _nlist_plugin.NeighborListBinnedSeg
        else:
            nlist_cls = _nlist_plugin.NeighborListGPUBinnedSeg
        self._cpp_obj = nlist_cls(self._simulation.state._cpp_sys_def,
                                  self.buffer, self.segments)

        super()._attach()