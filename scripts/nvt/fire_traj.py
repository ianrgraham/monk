import pathlib
import tempfile
import argparse
import sys
import gsd.hoomd

from inspect import signature

import hoomd
import numpy as np

from monk import prep, pair

valid_input_formats = [".gsd"]
valid_output_formats = [".gsd"]

parser = argparse.ArgumentParser(description="From a traj file, quench each frame to the inherent structure using FIRE")
parser.add_argument("ifile", type=str, help=f"Input file (allowed formats: {valid_input_formats}")
parser.add_argument("ofile", type=str, help=f"Output file (allowed formats: {valid_output_formats}")
parser.add_argument("--pair", nargs="+", help="Set the potential pair with any function callable in 'monk.pair'", default=["KA_LJ"])
parser.add_argument("--dt", type=float, default=2.5e-2, help="Timestep size")
parser.add_argument("--fire-steps", type=int, default=1000, help="Step interval to check for FIRE convergence")

args = parser.parse_args()

ifile = pathlib.Path(args.ifile)
ofile = pathlib.Path(args.ofile)

dt = args.dt
fire_steps = args.fire_steps

throw_away = int(args.throw_away / dt)

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
device = hoomd.device.GPU()
sim = hoomd.Simulation(device=device)
print(f"Running on {device.devices[0]}")
sim.create_state_from_gsd(str(ifile), frame=0)
traj = gsd.hoomd.open(str(ifile))


# set potential
print(f"Set potential. {{ pair: {pair_name}, args: {arguments} }}")
integrator = hoomd.md.minimize.FIRE(dt=dt)

cell = hoomd.md.nlist.Cell()
pot_pair = pair_func(cell, *arguments)
integrator.forces.append(pot_pair)

# start and end of ramp
nvt = hoomd.md.methods.NVE(filter=hoomd.filter.All())
integrator.methods.append(nvt)
sim.operations.integrator = integrator

# iterate over traj frames
for snap in traj:

    # load in snap and reset FIRE minimizer
    sim.create_state_from_snapshot(snap)
    integrator.reset()

    # run until converged
    while not integrator.converged():
        sim.run(fire_steps)

    # dump output to file
    hoomd.write.GSD.write(
        sim, str(ofile), mode="ab",
        filter=hoomd.filter.All()
    )