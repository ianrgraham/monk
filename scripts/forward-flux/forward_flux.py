import pathlib
import argparse
import gsd.hoomd
import sys

from inspect import signature

import hoomd
from hoomd import custom
import numpy as np
from schmeud.dynamics import thermal

import pandas as pd

from monk import pair


class AsyncTrigger(hoomd.trigger.Trigger):
    """NOT async in the Rust or JavaScript way
    
    Trigger by calling the activate() method"""

    def __init__(self):
        self.async_trig = False
        hoomd.trigger.Trigger.__init__(self)

    def activate(self):
        self.async_trig = True

    def compute(self, timestep):
        out = self.async_trig
        self.async_trig = False
        return out

class RestartablePeriodicTrigger(hoomd.trigger.Trigger):

    def __init__(self, period):
        assert(period >= 1)
        self.period = period
        self.state = period - 1
        hoomd.trigger.Trigger.__init__(self)

    def reset(self):
        self.state = self.period - 1

    def compute(self, timestep):
        if self.state >= self.period - 1:
            self.state = 0
            return True
        else:
            self.state += 1
            return False


class UpdatePosThermalizeVel(hoomd.custom.Action):
    """Custom action to handle feeding in a new set of positions through a snapshot"""

    def __init__(self, temp, new_snap=None):
        self.temp = temp
        self.new_snap = new_snap

    def set_snap(self, new_snap):
        self.new_snap = new_snap

    def set_temp(self, temp):
        self.temp = temp

    def act(self, timestep):
        old_snap = self._state.get_snapshot()
        if old_snap.communicator.rank == 0:
            N = old_snap.particles.N
            new_velocity = np.zeros((N,3))
            for i in range(N):
                old_snap.particles.velocity[i] = new_velocity[i]
                old_snap.particles.position[i] = self.new_snap.particles.position[i]
        self._state.set_snapshot(old_snap)
        self._state.thermalize_particle_momenta(hoomd.filter.All(), self.temp)

class PastSnapshotsBuffer(hoomd.custom.Action):
    """Custom action to hold onto past simulation snapshots"""

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


valid_input_formats = [".gsd"]
valid_output_formats = [".parquet"]

parser = argparse.ArgumentParser(description="From a traj file and a specified frame run a number of realizations of that system and compute the phop for each particle")
parser.add_argument("ifile", type=str, help=f"Input file (allowed formats: {valid_input_formats}")
parser.add_argument("ofile", type=str, help=f"Output file (allowed formats: {valid_output_formats}")
parser.add_argument("idx", nargs="+", type=int, help=f"Index to use for traj. Supply as many as wanted")
parser.add_argument("--replicas", type=int, default=1000, help=f"Number of replicas to simulate")
parser.add_argument("--pair", nargs="+", help="Set the potential pair with any function callable in 'monk.pair'", default=["KA_LJ"])
parser.add_argument("--dt", type=float, default=2.5e-3, help="Timestep size")
parser.add_argument("--temp", type=float, default=1.5, help="Starting and ending temperatures")
parser.add_argument("--dump-rate", type=float, default=0.1, help="Rate at which to store snapshots")
parser.add_argument("--sim-time", type=int, default=10, help="Time to simulate for each replica")
parser.add_argument("--seed", type=int, help="Random seed to initialize the RNG")
parser.add_argument("--run-fire", action="store_true", help="Randomize particle velocities at the start")
# parser.add_argument("--calc-mean", action="store_true", help="Calculate mean phop instead of ")



args = parser.parse_args()

ifile = pathlib.Path(args.ifile)
ofile = pathlib.Path(args.ofile)

dt = args.dt
traj_idx = args.idx
replicas = args.replicas
temp = args.temp
dump_rate = args.dump_rate
sim_time = args.sim_time
seed = args.seed

steps_per_dump = int(dump_rate/dt)
total_steps = int(sim_time/dt)

print(steps_per_dump, total_steps)


run_fire = args.run_fire
if run_fire:
    sys.exit("FIRE is not implemented for this code.")

pair_len = len(args.pair)
assert(pair_len >= 1)
pair_name = args.pair[0]
pair_args = tuple()
if pair_len > 1:
    pair_args = tuple(args.pair[1:])

pair_func = getattr(pair, pair_name)
signatures = list(signature(pair_func).parameters.values())[1:]
arguments = []
for arg, sig in zip(pair_args, signatures):
    arguments.append(sig.annotation(arg))
arguments = tuple(arguments)

# initialize hoomd state
print("Initialize HOOMD simulation")
device = hoomd.device.auto_select()
sim = hoomd.Simulation(device=device, seed=seed)
print(f"Running on {device.devices[0]}")
sim.create_state_from_gsd(str(ifile), frame=traj_idx[0])
traj = gsd.hoomd.open(str(ifile))
snap = traj[traj_idx[0]]


# set potential
print(f"Set potential. {{ pair: {pair_name}, args: {arguments} }}")
integrator = hoomd.md.Integrator(dt=dt)
cell = hoomd.md.nlist.Cell()
pot_pair = pair_func(cell, *arguments)
integrator.forces.append(pot_pair)

# start and end of ramp
nvt = hoomd.md.methods.NVT(
    kT=temp,
    filter=hoomd.filter.All(),
    tau=0.5)
integrator.methods.append(nvt)
sim.operations.integrator = integrator

custom_updater = UpdatePosThermalizeVel(temp, new_snap=snap)
snap_buffer = PastSnapshotsBuffer()
reset_config_trig = AsyncTrigger()
snap_buffer_trig = RestartablePeriodicTrigger(steps_per_dump)

custom_op = hoomd.update.CustomUpdater(action=custom_updater,
                                       trigger=reset_config_trig)

another_op = hoomd.update.CustomUpdater(action=snap_buffer,
                                        trigger=snap_buffer_trig)

sim.operations.add(custom_op)
sim.operations.add(another_op)

all_phops = []

for jdx in traj_idx:

    print(traj, jdx)

    custom_updater.set_snap(traj[jdx])

    # iterate over traj frames
    for idx in range(replicas):
        if idx%100 == 0:
            print("Frame", jdx, "\n--> Replica", idx)
        reset_config_trig.activate()
        snap_buffer_trig.reset()
        sim.run(2)

        sim.run(total_steps)

        # process phop
        mtraj = snap_buffer.get_snapshots()
        N = len(mtraj)
        phop = thermal.calc_phop(mtraj, tr_frames=N-1)

        all_phops.append([np.uint16(jdx), np.arange(len(phop[0]), dtype=np.uint16), np.uint16(idx), phop[0].astype(np.float32)]) # trivial unwrapping

        snap_buffer.clear()

df = pd.DataFrame(all_phops, columns=["frame", "id", "replica", "phop"]).explode(["id", "phop"])

df.to_parquet(ofile)