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