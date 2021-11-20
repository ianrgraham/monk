import pathlib
import tempfile
import argparse
import sys

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
parser.add_argument("--dt", type=float, default=2.5e-3, help="Timestep size")

args = parser.parse_args()

ifile = pathlib.Path(args.ifile)
ofile = pathlib.Path(args.ofile)

temp = tuple(args.temp)
dt = args.dt
seed = args.seed

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
sim = hoomd.Simulation(device=device, seed=seed)
print(f"Running on {device.devices[0]}")
sim.create_state_from_gsd(str(ifile), frame=ifile_index)


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

if args.rand_veloc:
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=temp)

sim.run(throw_away)

sim_steps = int(args.sim_time / dt)

thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamic_properties)

logger = hoomd.logging.Logger()

logger.add(thermodynamic_properties)
logger.add(sim, quantities=['timestep', 'walltime'])

dump_steps = int(args.dump_rate/dt)

gsd_writer = hoomd.write.GSD(filename=str(ofile),
                            trigger=hoomd.trigger.Periodic(dump_steps),
                            mode='wb',
                            filter=hoomd.filter.All(),
                            log = logger)

sim.operations.writers.append(gsd_writer)
gsd_writer.log = logger

sim.run(sim_steps)
