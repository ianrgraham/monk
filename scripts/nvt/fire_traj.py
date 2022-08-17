import pathlib
import argparse
import gsd.hoomd
import sys

from inspect import signature

import hoomd
from hoomd import custom
import numpy as np

from monk import pair, prep


class AsyncTrigger(hoomd.trigger.Trigger):

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

    def __init__(self, new_snap=None):
        self.new_snap = new_snap

    def set_snap(self, new_snap):
        self.new_snap = new_snap

    def act(self, timestep):
        old_snap = self._state.get_snapshot()
        # print("Worked!")
        if old_snap.communicator.rank == 0:
            N = old_snap.particles.N
            new_velocity = np.zeros((N, 3))
            for i in range(N):
                old_snap.particles.velocity[i] = new_velocity[i]
                old_snap.particles.position[
                    i] = self.new_snap.particles.position[i]
        self._state.set_snapshot(old_snap)


valid_input_formats = [".gsd"]
valid_output_formats = [".gsd"]

parser = argparse.ArgumentParser(
    description=
    "From a traj file, quench each frame to the inherent structure using FIRE")
parser.add_argument("ifile",
                    type=str,
                    help=f"Input file (allowed formats: {valid_input_formats}")
parser.add_argument(
    "ofile",
    type=str,
    help=f"Output file (allowed formats: {valid_output_formats}")
parser.add_argument(
    "--pair",
    nargs="+",
    help="Set the potential pair with any function callable in 'monk.pair'",
    default=["KA_LJ"])
parser.add_argument("--dt", type=float, default=2.5e-2, help="Timestep size")
parser.add_argument("--fire-steps",
                    type=int,
                    default=1000,
                    help="Step interval to check for FIRE convergence")
parser.add_argument(
    "--log-pair",
    action="store_true",
    help="Additionally log information about forces and energy")

args = parser.parse_args()

ifile = pathlib.Path(args.ifile)
ofile = pathlib.Path(args.ofile)

dt = args.dt
fire_steps = args.fire_steps
log_pair = args.log_pair

pair_func, arguments = prep.search_for_pair(args.pair)
pair_name: str = args.pair[0]

# initialize hoomd state
print("Initialize HOOMD simulation")
device = hoomd.device.auto_select()
sim = hoomd.Simulation(device=device)
print(f"Running on {device.devices[0]}")
sim.create_state_from_gsd(str(ifile), frame=0)
traj = gsd.hoomd.open(str(ifile))

# set potential
print(f"Set potential. {{ pair: {pair_name}, args: {arguments} }}")
integrator = hoomd.md.minimize.FIRE(dt, 1e-5, 1e-5, 1e-5)

cell = hoomd.md.nlist.Tree(buffer=0.2)
pot_pair = pair_func(cell, *arguments)
integrator.forces.append(pot_pair)

# start and end of ramp
nvt = hoomd.md.methods.NVE(filter=hoomd.filter.All())
integrator.methods.append(nvt)
sim.operations.integrator = integrator

custom_updater = UpdatePosZeroVel()
async_trig = AsyncTrigger()
async_write_trig = AsyncTrigger()

custom_op = hoomd.update.CustomUpdater(action=custom_updater,
                                       trigger=async_trig)

if log_pair:
    logger = hoomd.logging.Logger()
    logger.add(pot_pair, quantities=['energies', 'forces'])
else:
    logger = None

gsd_writer = hoomd.write.GSD(filename=str(ofile),
                             trigger=async_write_trig,
                             mode='wb',
                             filter=hoomd.filter.All(),
                             log=logger)

sim.operations.add(custom_op)
sim.operations.writers.append(gsd_writer)

# iterate over traj frames
for idx, snap in enumerate(traj):

    custom_updater.set_snap(snap)
    async_trig.activate()
    sim.run(2)
    integrator.reset()

    while not integrator.converged:
        sim.run(fire_steps)

    async_write_trig.activate()
    sim.run(2)
