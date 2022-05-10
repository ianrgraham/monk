from typing import Optional
import hoomd

from numpy import isin
import numpy as np
from hoomd.forward_flux import _forward_flux

class ForwardFluxSimulation(hoomd.Simulation):

    def __init__(self, device, pid, seed=None, fire_kwargs=None):
        super().__init__(device, seed=seed)
        if fire_kwargs is None:
            self.fire_kwargs = {"dt": 1e-2,
                                "force_tol":1e-2,
                                "angmom_tol":1e-2,
                                "energy_tol":1e-7}
        elif isinstance(fire_kwargs, dict) and "dt" in fire_kwargs:
            self.fire_kwargs = fire_kwargs
        else:
            raise ValueError("fire_kwargs")

        self.FIRE_STEPS = 100
        # self._basin_result = None
        self._ref_snap = None
        self._basin_barrier = None
        self._pid = pid

    # @property
    # def op(self, op: Callable[..., float]):
    #     self._op = op

    def _init_system(self, step):
        """Initialize the system State.

        Perform additional initialization operations not in the State
        constructor.
        """
        self._cpp_sys = _forward_flux.FFSystem(self.state._cpp_sys_def, step, self._pid)

        if self._seed is not None:
            self._state._cpp_sys_def.setSeed(self._seed)

        self._init_communicator()

    def sample_basin(self, steps: int, period: int, thermalize: Optional[int] = None, override_fire=None):
        
        self._assert_ff_state()

        # quench to the inherent structure
        self._run_fire(fire_kwargs=override_fire)
        self._ref_snap = self.state.get_snapshot()  # set python snapshot (in case)
        self._cpp_sys.setRefSnapFromPython(self._ref_snap._cpp_obj)  # set c++ snapshot for fast processing

        # possibly thermalize system
        if thermalize is not None:
            self._cpp_sys.sampleBasinForwardFluxes(thermalize)

        # collect a sampling of the order parameter around the basin
        basin_result = self._cpp_sys.sampleBasin(steps, period)

        # set the state back to the inherent structure
        self.state.set_snapshot(self._ref_snap)

        # use this to do some external processing and set "basin_barrier"
        return basin_result

    @property
    def basin_barrier(self):
        if self._basin_barrier is None:
            raise RuntimeError("basin_barrier must be set before access")
        return self._basin_barrier

    @basin_barrier.setter
    def basin_barrier(self, value):
        assert isinstance(value, float)
        self._cpp_sys.setBasinBarrier(value)
        self._basin_barrier = value
        

    def reset_state(self):
        if self._ref_snap is None:
            raise RuntimeError("The reference snapshot for the basin is not set")
        else:
            self._cpp_sys.setRefSnapFromPython(self._ref_snap._cpp_obj)


    def _assert_langevin(self):
        integrator = self.operations.integrator
        assert isinstance(integrator, hoomd.md.Integrator)
        assert len(integrator.constraints) == 0
        methods = integrator.methods
        assert len(methods) == 1
        langevin = methods[0]
        # assert isinstance(langevin, hoomd.md.methods.Langevin)
        return integrator, langevin, integrator.forces

    def _run_fire(self, fire_kwargs=None):

        # get integrator and associated data
        integrator, langevin, forces = self._assert_langevin()

        # build minimizer
        if fire_kwargs is not None:
            fire = hoomd.md.minimize.FIRE(fire_kwargs)
        else:
            fire = hoomd.md.minimize.FIRE(**self.fire_kwargs)
        nve = hoomd.md.methods.NVE(hoomd.filter.All())
        fire.methods = [nve]
        fire.forces = [forces.pop() for _ in range(len(forces))]

        self.operations.integrator = fire

        while not fire.converged:
            self.run(self.FIRE_STEPS)

        integrator.forces = [fire.forces.pop() for _ in range(len(fire.forces))]
        self.operations.integrator = integrator

        del nve
        del fire

    def _assert_ff_state(self):

        if not hasattr(self, '_cpp_sys'):
            raise RuntimeError('Cannot run before state is set.')
        if self._state._in_context_manager:
            raise RuntimeError(
                "Cannot call run inside of a local snapshot context manager.")
        if not self.operations._scheduled:
            self.operations._schedule()
        # if not hasattr(self, '_basin_result'):
        #     raise RuntimeError('Cannot run before basin has been established.')
        
        # warn about operations that will not be run during FFS
        # if len(self.operations.updaters) != 0:
        #     raise Warning("There are updaters loaded into the simulation, \
        #         They will not be run during forward flux sampling!")

        # if len(self.operations.computes) != 0:
        #     raise Warning("There are computes loaded into the simulation, \
        #         They will not be run during forward flux sampling!")

        # if len(self.operations.writers) != 0:
        #     raise Warning("There are writers loaded into the simulation, \
        #         They will not be run during forward flux sampling!")

    def thermalize_state(self, kT):
        self.state.thermalize_particle_momenta(hoomd.filter.All(), kT)

    def run_ff(self, basin_steps, trials=2000, thresh=0.99, barrier_step=0.01, thermalize: Optional[int] = None):
        
        self._assert_ff_state()

        # restrict simulation to MD methods w/ a Langevin thermostat (temporary)
        integrator, _, _ = self._assert_langevin()
        dt = integrator.dt

        # possibly thermalize system
        if thermalize is not None:
            self._cpp_sys.sampleBasinForwardFluxes(thermalize)

        # resample the basin, looking for crossings, and resetting when necessary
        
        a_crossings = self._cpp_sys.sampleBasinForwardFluxes(basin_steps)

        base_rate = len(a_crossings)/(basin_steps*dt)

        barriers = [self.basin_barrier + barrier_step]

        k_abs = []
        for state in a_crossings:

            last_trial_set = [state]
            rate = 1.0
            prod_trials = 1.0
            idx = 0
            last_rate = 0.0

            while True:
                if idx >= len(barriers):
                    barriers.append(barriers[idx-1] + barrier_step)
                barrier = barriers[idx]
                
                all_passed_trials = []
                k_trials = trials//len(last_trial_set)
                for new_state in last_trial_set:
                    for _ in range(k_trials):
                        opt_snap = self._cpp_sys.runFFTrial(barrier, new_state)
                        if opt_snap is not None:
                            all_passed_trials.append(opt_snap)

                last_rate = len(all_passed_trials)/(k_trials*len(last_trial_set))
                last_trial_set = all_passed_trials
                prod_trials *= float(k_trials)

                rate = len(all_passed_trials)/prod_trials
                
                if len(last_trial_set) == 0:
                    break
                elif last_rate >= thresh:
                    break

                idx += 1
            
            k_abs.append(rate)

        return base_rate*np.mean(k_abs)
                


