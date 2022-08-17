from typing import Callable, Optional
import hoomd
import numpy as np

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
        assert(period >= 1)
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
            new_velocity = np.zeros((N,3))
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
            case 0: # xy
                xy = box.xy + self._gamma * time_diff
                if xy > 0.5:
                    xy -= 1.0
                elif xy < -0.5:
                    xy += 1.0
                box.xy = xy
            case 1: # yz
                yz = box.yz + self._gamma * time_diff
                if yz > 0.5:
                    yz -= 1.0
                elif yz < -0.5:
                    yz += 1.0
                box.yz = yz
            case 2: # xz
                xz = box.xz + self._gamma * time_diff
                if xz > 0.5:
                    xz -= 1.0
                elif xz < -0.5:
                    xz += 1.0
                box.xz = xz
        self._state.set_box(box)
