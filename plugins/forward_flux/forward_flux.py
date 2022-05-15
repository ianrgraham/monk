from typing import Optional
from typing_extensions import Self
import freud.box
import hoomd

from numpy import isin
import numpy as np
import numpy.linalg as la
from numba import njit

from hoomd.forward_flux import _forward_flux
from hoomd import _hoomd
from hoomd.data.typeconverter import NDArrayValidator
from hoomd.data.parameterdicts import ParameterDict

from hoomd.custom import Action
from hoomd.update import CustomUpdater
from hoomd.operation import Updater

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
        self._ref_snap = None
        self._basin_barrier = None
        self._pid = pid

    def _init_system(self, step):
        """Initialize the system State.

        Perform additional initialization operations not in the State
        constructor.
        """
        self._cpp_sys = _forward_flux.FFSystem(self.state._cpp_sys_def, step, self._pid)

        if self._seed is not None:
            self._state._cpp_sys_def.setSeed(self._seed)

        self._init_communicator()

    def sample_basin(self, steps: int, period: int, thermalize: Optional[int] = None, override_fire=None, forces=None):
        
        self._assert_ff_state()

        # quench to the inherent structure
        self._run_fire(fire_kwargs=override_fire, sup_forces=forces)
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

    def _run_fire(self, fire_kwargs=None, sup_forces=None):

        # get integrator and associated data
        integrator, langevin, forces = self._assert_langevin()

        # build minimizer
        if fire_kwargs is not None:
            fire = hoomd.md.minimize.FIRE(fire_kwargs)
        else:
            fire = hoomd.md.minimize.FIRE(**self.fire_kwargs)
        nve = hoomd.md.methods.NVE(hoomd.filter.All())
        fire.methods = [nve]
        if sup_forces is None:
            fire.forces = [forces.pop() for _ in range(len(forces))]
        else:
            fire.forces = sup_forces

        self.operations.integrator = fire

        while not fire.converged:
            self.run(self.FIRE_STEPS)

        if sup_forces is None:  
            integrator.forces = [fire.forces.pop() for _ in range(len(fire.forces))]
        self.operations.integrator = integrator

        with self._state.cpu_local_snapshot as data:
            data.particles.velocity[:] = 0.0

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

    def thermalize_state(self, kT):
        self.state.thermalize_particle_momenta(hoomd.filter.All(), kT)

    def run_ff(self, basin_steps, trials=2000, thresh=0.95, barrier_step=0.01, op_thresh=None, thermalize: Optional[int] = None):

        old_seed = self.seed
        
        self._assert_ff_state()

        # restrict simulation to MD methods w/ a Langevin thermostat (temporary)
        integrator, _, _ = self._assert_langevin()
        dt = integrator.dt

        # possibly thermalize system
        if thermalize is not None:
            self._cpp_sys.sampleBasinForwardFluxes(thermalize)

        # resample the basin, looking for crossings, and resetting when necessary
        
        a_crossings = self._cpp_sys.sampleBasinForwardFluxes(basin_steps)

        print("Crossing:", len(a_crossings))

        base_rate = len(a_crossings)/(basin_steps*dt)

        barriers = [self.basin_barrier + barrier_step]

        k_abs = []
        for state in a_crossings:

            print("state:", len(k_abs))


            last_trial_set = [state]
            rate = 1.0
            prod_trials = 1.0
            idx = 0
            last_rate = 0.0

            while True:
                if idx >= len(barriers):
                    barriers.append(barriers[idx-1] + barrier_step)
                barrier = barriers[idx]

                # print("    bar:", barrier)
                
                all_passed_trials = []
                k_trials = trials//len(last_trial_set)
                for new_state in last_trial_set:
                    for k in range(k_trials):
                        self.seed += 1
                        opt_snap = self._cpp_sys.runFFTrial(barrier, new_state)
                        print(k, opt_snap)
                        if opt_snap is not None:
                            all_passed_trials.append(opt_snap)

                last_rate = len(all_passed_trials)/(k_trials*len(last_trial_set))
                last_trial_set = all_passed_trials
                prod_trials *= float(k_trials)

                rate = len(all_passed_trials)/prod_trials

                print(len(last_trial_set), last_rate)
                
                if len(last_trial_set) == 0:
                    break
                elif last_rate >= thresh and barrier >= op_thresh:
                    break

                idx += 1
            
            k_abs.append(rate)

        self.seed = old_seed

        return base_rate*np.mean(k_abs)


@njit
def _diff_with_rtag(ref_pos, pos, rtags):
    out = np.zeros_like(pos)
    n = len(rtags)
    for tag_idx in range(n):
        idx = rtags[tag_idx]
        out[idx] = pos[idx] - ref_pos[tag_idx]
    return out

class ZeroDrift(Action):

    def __init__(self, reference_positions, box):
        self._ref_pos = reference_positions
        self._box = freud.box.Box.from_box(box)
        self._imgs = np.array([0.0, 0.0, 0.0])

    @classmethod
    def from_state(cls, state: hoomd.State):
        return cls(state.get_snapshot().particles.position, state.box)

    def act(self, timestep):
        with self._state.cpu_local_snapshot as data:
            pos = data.particles.position._coerce_to_ndarray()
            rtags = data.particles.rtag._coerce_to_ndarray()
            diff = self._box.unwrap(_diff_with_rtag(self._ref_pos, pos, rtags), self._imgs)
            dx = np.mean(diff, axis=0)
            # print(dx)
            data.particles.position = self._box.unwrap(data.particles.position - dx, self._imgs)
