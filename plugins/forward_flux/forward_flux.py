from ctypes import Union
from msilib.schema import Error
from typing import Callable
from hoomd.md import _md
import hoomd
from hoomd.md.methods import Method
from hoomd.operation import _HOOMDBaseObject
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyTypes, OnlyIf, to_type_converter
from hoomd.filter import ParticleFilter
from hoomd.variant import Variant
from collections.abc import Sequence

from numpy import isin
from hoomd.forward_flux import _forward_flux

class ForwardFluxSimulation(hoomd.Simulation):

    def __init__(self, device, seed=None, fire_kwargs=None):
        super(ForwardFluxSimulation).__init__(device, seed=seed)
        if fire_kwargs is None:
            self.fire_kwargs = {"dt": 1e-2}
        elif isinstance(fire_kwargs, dict) and "dt" in fire_kwargs:
            self.fire_kwargs = fire_kwargs
        else:
            raise ValueError("fire_kwargs")

        self.FIRE_STEPS = 100

    @property
    def op(self, op: Callable[[hoomd.State], float]):
        self._op = op

    def _init_system(self, step):
        """Initialize the system State.

        Perform additional initialization operations not in the State
        constructor.
        """
        self._cpp_sys = _forward_flux.FFSystem(self.state._cpp_sys_def, step)

        if self._seed is not None:
            self._state._cpp_sys_def.setSeed(self._seed)

        self._init_communicator()

    def sample_basin(self):
        
        self._assert_ff_state()

        # quench to the inherint structure
        self._run_fire()


    def _assert_langevin(self):
        integrator = self.operations.integrator
        assert isinstance(integrator, hoomd.md.Integrator)
        assert len(integrator.constraints) == 0
        methods = integrator.methods
        assert len(methods) == 1
        langevin = methods[0]
        assert isinstance(langevin, hoomd.md.methods.Langevin)
        return integrator, langevin, integrator.forces

    def _run_fire(self):

        # get integrator and associated data
        integrator, langevin, forces = self._assert_langevin()

        # build minimizer
        fire = hoomd.md.minimize.FIRE(**self.fire_kwargs)
        fire.methods = [langevin]
        fire.forces = forces

        self.operations.integrator = fire

        while(not fire.converged()):
            self.run(self.FIRE_STEPS)

    def _assert_ff_state(self):

        if not hasattr(self, '_cpp_sys'):
            raise RuntimeError('Cannot run before state is set.')
        if self._state._in_context_manager:
            raise RuntimeError(
                "Cannot call run inside of a local snapshot context manager.")
        if not self.operations._scheduled:
            self.operations._schedule()
        if not hasattr(self, '_basin_result'):
            raise RuntimeError('Cannot run before basin has been established.')
        
        # warn about operations that will not be run during FFS
        if len(self.operations.updaters) != 0:
            raise Warning("There are updaters loaded into the simulation, \
                They will not be run during forward flux sampling")

        if len(self.operations.computes) != 0:
            raise Warning("There are computes loaded into the simulation, \
                They will not be run during forward flux sampling")

        if len(self.operations.writers) != 0:
            raise Warning("There are computes loaded into the simulation, \
                They will not be run during forward flux sampling")

    def run_ff(self):
        
        self._assert_ff_state()

        # restrict simulation to MD methods w/ a Langevin thermostat (temporary)
        self._assert_langevin()

        # resample the basin, looking for crossings, and resetting when necessary
        

        self._cpp_sys.runFFTrial()
