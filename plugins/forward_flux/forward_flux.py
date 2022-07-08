from typing import List, Optional, Tuple
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
    """Simulation that implements forward flux calculations
    
    Args:
        device (`hoomd.device.Device`): Device to execute the simulation.
        pid (int): Particle ID which will be appied during FFS calculations.
        seed (int): Random number seed.
        fire_kwargs (dict): Dictionary of parameters applied when quenching the
        system to it's inherint structure.

    `seed` sets the seed for the random number generator used by all operations
    added to this `Simulation`.

    Newly initialized `Simulation` objects have no state. Call
    `create_state_from_gsd` or `create_state_from_snapshot` to initialize the
    simulation's `state`.
    """

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

        Performs additional initialization operations not found in the State
        constructor.
        """
        self._cpp_sys = _forward_flux.FFSystem(self.state._cpp_sys_def, step, self._pid)

        if self._seed is not None:
            self._state._cpp_sys_def.setSeed(self._seed)

        self._init_communicator()

    @property
    def pid(self):
        """int: Particle ID of interest when calculating the order parameter."""
        return self._pid

    @pid.setter
    def pid(self, value):
        assert isinstance(value, int)
        self._pid = value
        self._cpp_sys.setPID(self._pid)

    def get_mapped_pid(self):
        return self._cpp_sys.getMappedPID()

    def _conf_space_distance(self, snap_i: hoomd.Snapshot, snap_j: hoomd.Snapshot) -> float:

        box = freud.box.Box.from_box(snap_i.configuration.box)

        # pos_i = box.unwrap(snap_i.particles.position, snap_i.particles.image)
        # pos_j = box.unwrap(snap_j.particles.position, snap_j.particles.image)

        dr = box.wrap(snap_j.particles.position - snap_i.particles.position)

        return np.linalg.norm(dr)

    def sample_all_sub_basins(
        self, 
        steps: int, 
        period: int,
        check_interval: int = 1000,
        reset_if_basin_left: bool = True,
        conf_space_cutoff: float = 0.4,
        thermalize: Optional[int] = None,
        override_fire=None, 
        forces=None
    ):

        self._assert_ff_state()

        # quench to the inherent structure
        self._run_fire(fire_kwargs=override_fire, sup_forces=forces)
        snap = self.state.get_snapshot()
        snap.particles.velocity[:] *= 0
        self._ref_snap = snap  # set python snapshot (in case)
        self._cpp_sys.setRefSnapFromPython(self._ref_snap._cpp_obj)  # set c++ snapshot for fast processing
        if thermalize is not None:
            self.state.thermalize_particle_momenta(hoomd.filter.All(), thermalize)

        output_basins = []
        valid_basins = []
        conf_dists = []

        start_step = self.timestep

        no_trigger = hoomd.trigger.On(0)

        write_triggers = [w.trigger for w in self.operations.writers]

        # sample the local basin, checkin every so often that we haven't jumped to another basin
        for i in range(steps//check_interval):
            basin_result = self._cpp_sys.sampleAllBasins(check_interval, period)
            output_basins.append(basin_result)

            snap = self.state.get_snapshot()
            for w in self.operations.writers:
                w.trigger = no_trigger
            self._run_fire(fire_kwargs=override_fire, sup_forces=forces)
            inher_struc_snap = self.state.get_snapshot()
            dist = self._conf_space_distance(inher_struc_snap, self._ref_snap)
            conf_dists.append(dist)

            is_valid_basin = dist < conf_space_cutoff
            if reset_if_basin_left and not is_valid_basin:
                self.state.set_snapshot(self._ref_snap)
                if thermalize is not None:
                    self.state.thermalize_particle_momenta(hoomd.filter.All(), thermalize)
                valid_basins.append(is_valid_basin)
            else:
                self.state.set_snapshot(snap)
                valid_basins.append(is_valid_basin)

            # reattach triggers once done
            for w, t in zip(self.operations.writers, write_triggers):
                w.trigger = t

        # set the state back to the inherent structure
        self.state.set_snapshot(self._ref_snap)

        return output_basins, valid_basins, conf_dists

    def sample_basin(
        self, 
        steps: int, 
        period: int, 
        thermalize: Optional[int] = None, 
        override_fire=None, 
        forces=None
    ):
        
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
        """float: barrier defining the edge of the originating basin."""
        if self._basin_barrier is None:
            raise RuntimeError("basin_barrier must be set before access")
        return self._basin_barrier

    @basin_barrier.setter
    def basin_barrier(self, value):
        assert isinstance(value, float)
        self._cpp_sys.setBasinBarrier(value)
        self._basin_barrier = value
        

    def reset_state(self):
        """Resets the simulation state back to the reference snapshot"""
        if self._ref_snap is None:
            raise RuntimeError("The reference snapshot for the basin is not set")
        else:
            self.state.set_snapshot(self._ref_snap)


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

        # run fire until convergence is reached
        while not fire.converged:
            self.run(self.FIRE_STEPS)

        # return simulation back to initial configuration
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

    def run_ff(
        self,
        basin_steps: int,
        collect: Optional[int] = None,
        sampling_thresh: Optional[float] = None,
        trials: int = 500,
        thresh: float = 0.90,
        barrier_step: float = 0.01,
        flex_step: Optional[float] = None,
        op_thresh: Optional[float] = None,
        floor: Optional[float] = None,
        attempts: int = 1,
        thermalize: Optional[int] = None,
        target_rate: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[float, List]:
        """run forward flux sampling for t"""

        # set boundary to reset state if particle goes too far during sampling
        if sampling_thresh is None:
            sampling_thresh = self._basin_barrier
        if sampling_thresh is None:
            raise RuntimeError
        
        # restrict simulation to MD methods w/ a Langevin thermostat (temporary)
        self._assert_ff_state()
        integrator, _, _ = self._assert_langevin()
        dt = integrator.dt

        # possibly thermalize system
        if thermalize is not None:
            self._cpp_sys.run(thermalize, False)

        # resample the basin, looking for crossings, and resetting when necessary
        a_crossings = self._cpp_sys.sampleBasinForwardFluxes(basin_steps)

        # optional resample to collect enough states
        if collect is not None:
            while len(a_crossings) < collect:
                if self._cpp_sys.computeOrderParameter() > sampling_thresh:
                    self.reset_state()
                a_crossings.extend(self._cpp_sys.sampleBasinForwardFluxes(basin_steps))

        if verbose:
            print("A barrier crossings:", len(a_crossings))

        if target_rate is not None:
            assert target_rate > 0.0 and target_rate <= 1.0
            inv_target_rate = 1/target_rate
        else:
            inv_target_rate = 2.0


        base_rate = len(a_crossings)/(basin_steps*dt)

        # main forward flux calculation loop
        k_abs = []
        barriers = []
        rates = []
        for state in a_crossings:
            if verbose:
                print("state:", len(k_abs))
            barrier = self.basin_barrier
            state_barriers = []
            state_rates = []

            last_trial_set = [state]
            rate = 1.0
            prod_trials = 1.0
            idx = 0
            last_rate = 1.0

            finish = None

            if flex_step is not None:
                last_step = barrier_step

            while True:
                old_barrier = barrier
                attempt_idx = 0
                while attempt_idx < attempts:
                    if flex_step is not None:
                        next_step = max(min(inv_target_rate*last_rate*last_step, barrier_step), flex_step)
                        barrier += next_step
                        last_step = next_step
                    else:
                        barrier += barrier_step
                    
                    all_passed_trials = []
                    k_trials = trials//len(last_trial_set)
                    for new_state in last_trial_set:
                        for k in range(k_trials):
                            opt_snap, _ = self._cpp_sys.runFFTrial(barrier, new_state, False)
                            if opt_snap is not None:
                                all_passed_trials.append(opt_snap)

                    last_rate = len(all_passed_trials)/(k_trials*len(last_trial_set))
                    attempt_idx += 1
                    if last_rate > 0.0:
                        break
                    barrier = old_barrier
                last_trial_set = all_passed_trials
                prod_trials *= float(k_trials)

                rate = len(all_passed_trials)/prod_trials
                if verbose:
                    print("barrier_idx:", idx, "|barrier_op:", barrier,  f"|last_rate: {last_rate}")

                state_barriers.append(barrier)
                state_rates.append(rate)
                
                if floor is not None and rate < floor:
                    rate = 0.0
                    break
                elif len(last_trial_set) == 0:
                    break
                elif barrier >= op_thresh:
                    if last_rate >= thresh:
                        if finish is None:
                            finish = barrier
                        elif barrier - finish >= barrier_step:
                            break
                    elif finish is not None:
                        finish = None

                idx += 1
            if verbose:
                print("final_rate:", rate)
            k_abs.append(rate)
            barriers.append(state_barriers)
            rates.append(state_rates)

        return base_rate*np.mean(k_abs), barriers, rates


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

    @classmethod
    def from_state(cls, state: hoomd.State):
        return cls(state.get_snapshot().particles.position, state.box)

    def act(self, timestep):
        with self._state.cpu_local_snapshot as data:
            pos = data.particles.position._coerce_to_ndarray()
            rtags = data.particles.rtag._coerce_to_ndarray()
            diff = self._box.wrap(_diff_with_rtag(self._ref_pos, pos, rtags))
            dx = np.mean(diff, axis=0)
            data.particles.position = self._box.wrap(data.particles.position - dx)