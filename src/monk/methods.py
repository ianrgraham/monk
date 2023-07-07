from typing import Callable, Optional
import hoomd
import gsd.hoomd
import numpy as np
import tqdm

import freud

class VerifyEquilibrium(hoomd.custom.Action):
    """Computes dynamics to verify equilibration."""

    def __init__(self, k: Optional[float] = None, max_alpha_time: float = 1e2):
        self.last_pos = None
        self.last_image = None
        self.last_tstep = None
        if k is None:
            self.k = 7.14
        else:
            self.k = k
        self.max_alpha_time = max_alpha_time
        self.alphas = []
        self.Ds = []
        self.msds = []
        self.sisfs = []
        self.tsteps = []
    
    def act(self, timestep):
        
        if self.last_pos is None:
            snap = self._state.get_snapshot()
            self.last_pos = snap.particles.position
            self.last_image = snap.particles.image
            self.first_tstep = timestep
            self.last_tstep = timestep
            self.last_msd = 0.0
            self.alpha_time = 0.0
            self.measure_D = False
            self.measured_D = 0.0
            self.measured_alpha = 0.0
        else:
            if self.measure_D:
                dt = self._state._simulation.operations.integrator.dt
                if (timestep - self.last_tstep)*dt > self.measured_alpha:
                    snap = self._state.get_snapshot()
                    pos = snap.particles.position
                    image = snap.particles.image
                    sim_box = self._state.box
                    if sim_box.is2D:
                        dim = 2
                    else:
                        dim = 3
                    box = freud.box.Box.from_box(sim_box)
                    unwrapped_pos = box.unwrap(pos, image - self.last_image)
                    msd = np.mean(np.sum(np.square(unwrapped_pos - self.last_pos), axis=-1))
                    self.msds.append(msd)
                    self.tsteps.append(timestep - self.first_tstep)
                    D = (msd - self.last_msd) / (timestep - self.last_tstep) / dt / (2 * dim)
                    self.measured_D = D
                    self.Ds.append(self.measured_D)
                    self.alphas.append(self.measured_alpha)
                    self.measure_D = False
                    self.alpha_time = 0.0
                    self.first_tstep = timestep
                    self.last_pos = snap.particles.position
                    self.last_image = snap.particles.image
            else:
                snap = self._state.get_snapshot()
                dt = self._state._simulation.operations.integrator.dt
                pos = snap.particles.position
                image = snap.particles.image
                sim_box = self._state.box
                if sim_box.is2D:
                    dim = 2
                else:
                    dim = 3
                box = freud.box.Box.from_box(sim_box)
                unwrapped_pos = box.unwrap(pos, image - self.last_image)
                msd = np.mean(np.sum(np.square(unwrapped_pos - self.last_pos), axis=-1))
                self.msds.append(msd)
                time_diff = timestep - self.first_tstep
                self.tsteps.append(time_diff)
                self.last_tstep = timestep
                self.last_msd = msd

                x = self.k * np.linalg.norm(pos - self.last_pos, axis=-1)
                sisf = np.mean(np.sin(x)/x)
                self.sisfs.append(sisf)

                # print(f"{D} {sisf}")

                self.alpha_time = (timestep - self.first_tstep) * dt
                if sisf < np.exp(-1.0):
                    self.measure_D = True
                    self.measured_alpha = self.alpha_time
                elif self.alpha_time > self.max_alpha_time:
                    raise RuntimeError("Alpha relaxation time is too long.")

class AsyncTrigger(hoomd.trigger.Trigger):
    """Used by FIRE quench method."""

    def __init__(self):
        self.async_trig = False
        hoomd.trigger.Trigger.__init__(self)

    def activate(self):
        self.async_trig = True

    def compute(self, timestep):
        out = self.async_trig
        # if out:
        #     print("Triggered")
        self.async_trig = False
        return out


class UpdatePosZeroVel(hoomd.custom.Action):
    """Used by FIRE quench method."""

    def __init__(self, new_snap=None):
        self.new_snap = new_snap

    def set_snap(self, new_snap):
        self.new_snap = new_snap

    def act(self, timestep):
        old_snap = self._state.get_snapshot()
        if old_snap.communicator.rank == 0:
            N = old_snap.particles.N
            old_snap.particles.velocity[:] = np.zeros((N, 3))
            old_snap.particles.position[:] = self.new_snap.particles.position
            old_snap.configuration.box = self.new_snap.configuration.box
        self._state.set_snapshot(old_snap)


class RemoteTrigger(hoomd.trigger.Trigger):
    """Trigger which is activated on the next timestep by calling the `trigger`
    method."""

    def __init__(self):
        hoomd.trigger.Trigger.__init__(self)
        self.trig = False

    def trigger(self):
        self.trig = True

    def compute(self, timestep: int):
        if self.trig:
            self.trig = False
            return True
        else:
            return False


class RestartablePeriodicTrigger(hoomd.trigger.Trigger):
    """Periodic trigger that can be reset."""

    def __init__(self, period: int, phase: Optional[int] = None):
        assert (period >= 1)
        hoomd.trigger.Trigger.__init__(self)
        self.period = period
        self.phase = phase
        self.reset()

    def reset(self):
        if self.phase is None:
            self.state = self.period - 1
        else:
            self.state = self.phase

    def compute(self, timestep: int):
        if self.state >= self.period - 1:
            self.state = 0
            return True
        else:
            self.state += 1
            return False


class UpdatePosThermalizeVel(hoomd.custom.Action):
    """"""

    def __init__(self, temp, snap=None):
        self.temp = temp
        self.snap = snap

    def set_snap(self, new_snap):
        self.snap = new_snap

    def set_temp(self, temp):
        self.temp = temp

    def act(self, timestep):
        old_snap = self._state.get_snapshot()
        if old_snap.communicator.rank == 0:
            N = old_snap.particles.N
            new_velocity = np.zeros((N, 3))
            for i in range(N):
                old_snap.particles.velocity[i] = new_velocity[i]
                old_snap.particles.position[i] = self.snap.particles.position[i]
        self._state.set_snapshot(old_snap)
        self._state.thermalize_particle_momenta(hoomd.filter.All(), self.temp)


class PastSnapshotsBuffer(hoomd.custom.Action):
    """Custom action to hold onto past simulation snapshots (in memory)."""

    def __init__(self):
        self.snap_buffer = []

    def clear(self):
        self.snap_buffer.clear()

    def get_snapshots(self):
        return self.snap_buffer

    def force_push(self):
        self.act(None)

    def act(self, timestep):
        snap = self._state.get_snapshot()
        self.snap_buffer.append(snap)


class LogTrigger(hoomd.trigger.Trigger):

    def __init__(self, base: int, start: float, step: float, ref_timestep: float):
        self.ref = ref_timestep
        self.base = base
        self.step = step
        self.cidx = start
        self.cstep = self.base**self.cidx
        super().__init__()

    def compute(self, timestep):
        result = False
        while timestep - self.ref > np.round(self.cstep):
            result = True
            self.cidx += self.step
            self.cstep = self.base**self.cidx
        return result


class NextTrigger(hoomd.trigger.Trigger):

    def __init__(self):
        self._next = False
        self._cached_tstep = None
        super().__init__()

    def next(self, ):
        self._next = True

    def compute(self, timestep):
        if self._next:
            self._next = False
            self._cached_tstep = timestep
            return True
        elif self._cached_tstep is not None and self._cached_tstep == timestep:
            return True
        else:
            return False


class ConstantShear(hoomd.custom.Action):
    """Apply a constant shear rate to the simulation box.

    Arguments
    ---------
    - gamma: float - the shear rate (in units of box ratio per time step)
    - timestep: int - the initial reference timestep
    - pair: str - the dimension pair to apply the shear to"""

    def __init__(self, gamma: float, timestep: int, pair: str = "xz"):
        super().__init__()
        self._gamma = gamma
        self._last_timestep = timestep
        match pair:
            case "xy":
                self._pair_mode = 0
            case "yz":
                self._pair_mode = 1
            case "xz":
                self._pair_mode = 2

    def act(self, timestep):
        time_diff = timestep - self._last_timestep
        self._last_timestep = timestep
        box = self._state.box
        match self._pair_mode:
            case 0:  # xy
                xy = box.xy + self._gamma * time_diff
                if xy > 0.5:
                    xy -= 1.0
                elif xy < -0.5:
                    xy += 1.0
                box.xy = xy
            case 1:  # yz
                yz = box.yz + self._gamma * time_diff
                if yz > 0.5:
                    yz -= 1.0
                elif yz < -0.5:
                    yz += 1.0
                box.yz = yz
            case 2:  # xz
                xz = box.xz + self._gamma * time_diff
                if xz > 0.5:
                    xz -= 1.0
                elif xz < -0.5:
                    xz += 1.0
                box.xz = xz
        self._state.set_box(box)


class KeepBoxTiltsSmall(hoomd.custom.Action):
    """Apply a correction to keep Box tilt factors with the range [-0.5, 0.5)."""

    def __init__(self):
        super().__init__()

    def act(self, timestep):
        box = self._state.box
        change = False
        
        if box.xy >= 0.5:
            box.xy -= 1.0
            change = True
        elif box.xy < -0.5:
            box.xy += 1.0
            change = True
        if box.yz >= 0.5:
            box.yz -= 1.0
            change = True
        elif box.yz < -0.5:
            box.yz += 1.0
            change = True
        if box.xz >= 0.5:
            box.xz -= 1.0
            change = True
        elif box.xz < -0.5:
            box.xz += 1.0
            change = True

        if change:
            self._state.set_box(box)

class SinusoidalShear(hoomd.custom.Action):
    """Apply a sinusoidal shear protocol."""

    def __init__(self, gamma: float, timestep: int, period: int, pair: str = "xy"):
        super().__init__()
        self._gamma = gamma
        self._start = timestep
        self._period = period
        match pair:
            case "xy":
                self._pair_mode = 0
            case "yz":
                self._pair_mode = 1
            case "xz":
                self._pair_mode = 2

    def act(self, timestep):
        box = self._state.box
        time_diff = (timestep - self._start) % self._period

        shear = np.sin(2 * np.pi * time_diff / self._period) * self._gamma

        match self._pair_mode:
            case 0:  # xy
                box.xy = shear
            case 1:  # yz
                box.yz = shear
            case 2:  # xz
                box.xz = shear
            
        self._state.set_box(box)

class SimpleViscousForce(hoomd.md.force.Custom):
    """A simple viscous force that acts on all particles."""

    def __init__(self, gamma: float):
        super().__init__(aniso=False)
        self.gamma = gamma

    def set_forces(self, timestep):
        with self._state.cpu_local_snapshot as snap:
            with self.cpu_local_force_arrays as arrays:
                arrays.force[:] = -snap.particles.velocity * self.gamma
            

class Oscillate(hoomd.variant.Variant):
    def __init__(self, amplitude: float, t_start: int, period: int, phase: float = 0):
        hoomd.variant.Variant.__init__(self)
        self._amp = amplitude / 2
        self._actual_amp = amplitude
        self._t_start = t_start
        self._period = period
        self._phase = phase

    def __call__(self, timestep):
        time = (timestep - self._t_start) % self._period
        val = (np.sin(2 * np.pi * time / self._period + self._phase) + 1) * self._amp
        # print(val)
        return val

    def _min(self):
        return 0

    def _max(self):
        return self._actual_amp
    
class OscillateCounter(hoomd.variant.Variant):
    def __init__(self, amplitude: float, period: int, phase: float = 0):
        hoomd.variant.Variant.__init__(self)
        self._amp = amplitude / 2
        self._actual_amp = amplitude
        self._t_start = 0
        self._count = 0
        self._period = period
        self._phase = phase

    def __call__(self, timestep):
        time = (self._count - self._t_start) % self._period
        val = (np.sin(2 * np.pi * time / self._period + self._phase) + 1) * self._amp
        # print(val)
        return val
    
    def inc(self):
        self._count += 1

    def _min(self):
        return 0

    def _max(self):
        return self._actual_amp

def fire_minimize_frames(
        sim: hoomd.Simulation,
        input_traj: gsd.hoomd.HOOMDTrajectory,
        out_file: str,
        fire_steps: int = 10_000
):
    """Generate a FIRE minimized trajectory from another."""

    if not isinstance(sim.operations.integrator, hoomd.md.minimize.FIRE):
        raise ValueError("Loaded integrator is not a FIRE minimizer.")

    fire = sim.operations.integrator

    custom_updater = UpdatePosZeroVel()
    async_trig = AsyncTrigger()
    async_write_trig = AsyncTrigger()

    custom_op = hoomd.update.CustomUpdater(action=custom_updater,
                                           trigger=async_trig)

    gsd_writer = hoomd.write.GSD(filename=str(out_file),
                                 trigger=async_write_trig,
                                 mode='wb',
                                 filter=hoomd.filter.All())

    sim.operations.add(custom_op)
    sim.operations.writers.append(gsd_writer)
    for idx, snap in tqdm.tqdm(enumerate(input_traj)):

        custom_updater.set_snap(snap)
        # print(snap.configuration.box)
        # print(snap.particles.position[32439])
        async_trig.activate()
        sim.run(2)
        fire.reset()
        # print("pre-run")
        while not fire.converged:
            sim.run(fire_steps)
        # print("post-run")
        async_write_trig.activate()
        sim.run(2)
