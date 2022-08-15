# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""HOOMD plugin for a variety of pair interactions."""

import copy
import warnings

from hoomd import _hoomd
from hoomd.logging import log
from hoomd.md import _md, force
from hoomd.pair_plugin import _pair_plugin
import hoomd
from hoomd.md.nlist import NeighborList
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import (OnlyFrom, OnlyTypes, nonnegative_real)
import hoomd.md.pair as _pair

validate_nlist = OnlyTypes(NeighborList)

class ModLJ(_pair.Pair):
    r"""Modified Lennard-Jones pair potential to showcase an example of a pair plugin.

    Args:
        nlist (`hoomd.md.nlist.NeighborList`): Neighbor list.
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
    _cpp_class_name = "PotentialPairMLJ"

    def __init__(self, nlist, default_r_cut=None, default_r_on=0., mode='none'):
        super().__init__(nlist, default_r_cut, default_r_on, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float, sigma=float, delta=float, len_keys=2))
        self._add_typeparam(params)

    def _attach(self):
        """Slightly modified with regard to the base class `md.Pair`.
        
        In particular, we search for `PotentialPairExample` in `hoomd.pair_plugin._pair_plugin`
        instead of `hoomd.md._md`.
        """
        # create the c++ mirror class
        if not self.nlist._added:
            self.nlist._add(self._simulation)
        else:
            if self._simulation != self.nlist._simulation:
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
        nlist (`hoomd.md.nlist.NeighborList`): Neighbor list.
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
        lj = pair.ExamplePair(nl, default_r_cut=1.0)
        lj.params[('A', 'A')] = {'sigma': 1.0, 'epsilon': 1.0}
        lj.r_cut[('A', 'A')] = 1.0
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
        
        In particular, we search for `PotentialPairHertzian` in `_pair_plugin`
        instead of `md.pair` as we would have done in the source code.
        """
        # create the c++ mirror class
        if not self.nlist._added:
            self.nlist._add(self._simulation)
        else:
            if self._simulation != self.nlist._simulation:
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


class HPFPair(force.Force):
    r"""Base class hard-particle frictional forces.

    Note:
        :py:class:`HPFPair` is the base class for all frictional pair forces. Users should
        not instantiate this class directly.

    .. py:attribute:: r_cut

        Cuttoff radius beyond which the energy and force are 0
        :math:`[\mathrm{length}]`. *Optional*: defaults to the value
        ``default_r_cut`` specified on construction.

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `float`])

    .. py:attribute:: mode

        *mode*, *optional*: defaults to ``"none"``.
        Possible values: ``"none"``, ``"shift"``

        Type: `str`

    .. py:attribute:: nlist

        Neighbor list used to compute the pair force.

        Type: `hoomd.md.nlist.NeighborList`
    """

    # The accepted modes for the potential. Should be reset by subclasses with
    # restricted modes.
    _accepted_modes = ("none",)

    @log(category="pair", requires_run=True)
    def nlist_pairs(self):
        pass

    @log(category="pair", requires_run=True)
    def pair_conserv_forces(self):
        pass

    @log(category="pair", requires_run=True)
    def pair_friction_forces(self):
        pass

    @log(category="pair", requires_run=True)
    def pair_torques(self):
        pass

    def __init__(self, nlist, default_r_cut=None, mode='none', mus=0.0, mur=0.0, ks=0.0, kr=0.0):
        super().__init__()
        tp_r_cut = TypeParameter(
            'r_cut', 'particle_types',
            TypeParameterDict(nonnegative_real, len_keys=2))
        if default_r_cut is not None:
            tp_r_cut.default = default_r_cut

        type_params = [tp_r_cut]

        self._extend_typeparam(type_params)
        self._param_dict.update(
            ParameterDict(mode=OnlyFrom(self._accepted_modes),
                          nlist=hoomd.md.nlist.NeighborList))
        self.mode = mode
        self.nlist = nlist

        self.mus = mus
        self.mur = mur
        self.ks = ks
        self.kr = kr

        

    def _add(self, simulation):
        super()._add(simulation)
        self._add_nlist()

    def _add_nlist(self):
        nlist = self.nlist
        deepcopy = False
        if not isinstance(self._simulation, hoomd.Simulation):
            if nlist._added:
                deepcopy = True
            else:
                return
        # We need to check if the force is added since if it is not then this is
        # being called by a SyncedList object and a disagreement between the
        # simulation and nlist._simulation is an error. If the force is added
        # then the nlist is compatible. We cannot just check the nlist's _added
        # property because _add is also called when the SyncedList is synced.
        if deepcopy or nlist._added and nlist._simulation != self._simulation:
            warnings.warn(
                f"{self} object is creating a new equivalent neighbor list."
                f" This is happending since the force is moving to a new "
                f"simulation. Set a new nlist to suppress this warning.",
                RuntimeWarning)
            self.nlist = copy.deepcopy(nlist)
        self.nlist._add(self._simulation)
        # This is ideopotent, but we need to ensure that if we change
        # neighbor list when not attached we handle correctly.
        self._add_dependency(self.nlist)

    def _attach(self):
        """Slightly modified with regard to the base class `md.Friction`.
        """
        # create the c++ mirror class
        if not self.nlist._added:
            self.nlist._add(self._simulation)
        else:
            if self._simulation != self.nlist._simulation:
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
            # TODO GPU version is not yet implemented
            print("GPU version not yet implemented")
            cls = getattr(_pair_plugin, self._cpp_class_name) #  + "GPU"
            self.nlist._cpp_obj.setStorageMode(
                _md.NeighborList.storageMode.full)
        self._cpp_obj = cls(self._simulation.state._cpp_sys_def,
                            self.nlist._cpp_obj,
                            self.mus,
                            self.mur,
                            self.ks,
                            self.kr)
        
        super()._attach()

    def _setattr_param(self, attr, value):
        if attr == "nlist":
            self._nlist_setter(value)
            return
        super()._setattr_param(attr, value)

    def _nlist_setter(self, new_nlist):
        if new_nlist is self.nlist:
            return
        if self._attached:
            raise RuntimeError("nlist cannot be set after scheduling.")
        old_nlist = self.nlist
        self._param_dict._dict["nlist"] = new_nlist
        if self._added:
            self._add_nlist()
            old_nlist._remove_dependent(self)


class HarmHPF(HPFPair):
    """Harmonic hard-particle interaction with frictional force"""

    _cpp_class_name = "PotentialPairHPF"

    def __init__(self, nlist, default_r_cut=None, mode='none', mus=0.0, mur=0.0, ks=0.0, kr=0.0):
        super().__init__(nlist, default_r_cut, mode, mus, mur, ks, kr)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(k=float, rcut=float, len_keys=2))
        self._add_typeparam(params)