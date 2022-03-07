# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Example Pair."""

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.pair_plugin import _pair_plugin
import hoomd
from hoomd.md.nlist import NList
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import (OnlyFrom, OnlyTypes, nonnegative_real)
import hoomd.md.pair as _pair

validate_nlist = OnlyTypes(NList)

class ModLJ(_pair.Pair):
    r"""Modified Lennard-Jones pair potential to showcase an example of a pair plugin.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    `ExampleLJ` specifies that a modified Lennard-Jones pair potential should be
    applied between every non-excluded particle pair in the simulation.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{LJ}}(r)  = & 4 \varepsilon \left[ \left(
        \frac{\sigma}{r-\Delta} \right)^{12} - \left( \frac{\sigma}{r-\Delta}
        \right)^{6} \right]; & r < r_{\mathrm{cut}} \\
        = & 0; & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    See `Pair` for details on how forces are calculated and the available
    energy shifting and smoothing modes.

    .. py:attribute:: params

        The example potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`
        * ``delta`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        lj = pair.ExamplePair(nl, default_r_cut=2.5)
        lj.params[('A', 'A')] = {'sigma': 1.0, 'epsilon': 1.0, 'delta': 0.25}
        lj.r_cut[('A', 'A')] = 2.5
    """

    # Name of the potential we want to reference on the C++ side
    _cpp_class_name = "PotentialPairExample"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float, sigma=float, delta=float, len_keys=2))
        self._add_typeparam(params)

    def _attach(self):
        """Slightly modified with regard to the base class `md.Pair`.
        
        In particular, we search for `PotentialPairExample` in `_pair_plugin`
        instead of `md.pair` as we would have done in the source code.
        """
        # create the c++ mirror class
        if not self._nlist._added:
            self._nlist._add(self._simulation)
        else:
            if self._simulation != self._nlist._simulation:
                raise RuntimeError("{} object's neighbor list is used in a "
                                   "different simulation.".format(type(self)))
        if not self.nlist._attached:
            self.nlist._attach()
        # Find definition of _cpp_class_name in _pair_plugin
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = getattr(_pair_plugin, self._cpp_class_name)
            self.nlist._cpp_obj.setStorageMode(
                _md.NeighborList.storageMode.half)
        else:
            cls = getattr(_pair_plugin, self._cpp_class_name + "GPU")
            self.nlist._cpp_obj.setStorageMode(
                _md.NeighborList.storageMode.full)
        self._cpp_obj = cls(self._simulation.state._cpp_sys_def,
                            self.nlist._cpp_obj)
        
        grandparent = super(_pair.Pair, self)
        grandparent._attach()


class Hertzian(_pair.Pair):
    r"""Hertzian pair potential.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        default_r_on (float): Default turn-on radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting/smoothing mode.

    See `Pair` for details on how forces are calculated and the available
    energy shifting and smoothing modes.

    .. py:attribute:: params

        The example potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) -
          energy parameter :math:`\varepsilon` :math:`[\mathrm{energy}]`
        * ``sigma`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`
        * ``delta`` (`float`, **required**) -
          particle size :math:`\sigma` :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        lj = pair.ExamplePair(nl, default_r_cut=2.5)
        lj.params[('A', 'A')] = {'sigma': 1.0, 'epsilon': 1.0, 'delta': 0.25}
        lj.r_cut[('A', 'A')] = 2.5
    """

    # Name of the potential we want to reference on the C++ side
    _cpp_class_name = "PotentialPairHertzian"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float, sigma=float, len_keys=2))
        self._add_typeparam(params)

    def _attach(self):
        """Slightly modified with regard to the base class `md.Pair`.
        
        In particular, we search for `PotentialPairExample` in `_pair_plugin`
        instead of `md.pair` as we would have done in the source code.
        """
        # create the c++ mirror class
        if not self._nlist._added:
            self._nlist._add(self._simulation)
        else:
            if self._simulation != self._nlist._simulation:
                raise RuntimeError("{} object's neighbor list is used in a "
                                   "different simulation.".format(type(self)))
        if not self.nlist._attached:
            self.nlist._attach()
        # Find definition of _cpp_class_name in _pair_plugin
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = getattr(_pair_plugin, self._cpp_class_name)
            self.nlist._cpp_obj.setStorageMode(
                _md.NeighborList.storageMode.half)
        else:
            cls = getattr(_pair_plugin, self._cpp_class_name + "GPU")
            self.nlist._cpp_obj.setStorageMode(
                _md.NeighborList.storageMode.full)
        self._cpp_obj = cls(self._simulation.state._cpp_sys_def,
                            self.nlist._cpp_obj)
        
        grandparent = super(_pair.Pair, self)
        grandparent._attach()
