import pathlib
import tempfile
import argparse
import sys
# import matplotlib.pyplot as plt

from inspect import signature

import hoomd
import numpy as np

from monk import prep, pair

valid_output_formats = [".gsd"]

parser = argparse.ArgumentParser(description="Initialize a packing of LJ-like"
        " particles, equilibrate them, and then quench the configuration.")
parser.add_argument("ofile", type=str, help=f"Output file (allowed formats: {valid_output_formats}")
parser.add_argument("--num", type=int, help="Number of particles to simulate.", default=256)
parser.add_argument("--dt", type=float, default=1e-3)
parser.add_argument("--phi", type=float, default=1.2)
parser.add_argument("--temp", type=float, default=1.5)
parser.add_argument("--deltas", type=float, nargs=3, default=[0.0, 0.5, 11])
parser.add_argument("--equil-time", type=int, default=100)
parser.add_argument("--step-time", type=float, default=100)
parser.add_argument("--dump-rate", type=float, default=0.01)
parser.add_argument("--seed", type=int, help="Random seed to initialize the RNG.", default=27)

args = parser.parse_args()

ofile = pathlib.Path(args.ofile)

N = args.num
temp = args.temp
dt = args.dt
phi = args.phi
seed = args.seed

delta_params = tuple(args.deltas)

# initialize hoomd state
print("Initialize HOOMD simulation")
device = hoomd.device.auto_select()
sim = hoomd.Simulation(device=device, seed=seed)
print(f"Running on {device.devices[0]}")

# create equilibrated 3D LJ configuration
rng = prep.init_rng(seed)  # random generator for placing particles
snapshot = prep.approx_euclidean_snapshot(
    N,
    np.cbrt(N / phi),
    rng,
    dim=3,
    particle_types=['A', 'B'],
    ratios=[80, 20])
sim.create_state_from_snapshot(snapshot)


# set simtential
print("Set potential")
integrator = hoomd.md.Integrator(dt=dt)
cell = hoomd.md.nlist.Cell()
pot_pair = pair.KA_modLJ(cell, 0.0)
integrator.forces.append(pot_pair)

# start and end of ramp
equil_steps = int(args.equil_time / dt)
delta_steps = int(args.step_time / dt)

nvt = hoomd.md.methods.NVT(
    kT=temp,
    filter=hoomd.filter.All(),
    tau=0.5)
integrator.methods.append(nvt)
sim.operations.integrator = integrator

sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=temp)

thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamic_properties)

logger = hoomd.logging.Logger()

logger.add(thermodynamic_properties)
logger.add(sim, quantities=['timestep', 'walltime'])

dump_time = int(args.dump_rate/dt)

gsd_writer = hoomd.write.GSD(filename=str(ofile),
                            trigger=hoomd.trigger.Periodic(dump_time),
                            mode='wb',
                            filter=hoomd.filter.All(),
                            log=logger)

sim.operations.writers.append(gsd_writer)

sim.run(equil_steps)

print(thermodynamic_properties.pressure)

for delta in np.linspace(*delta_params):
    
    lj = pair.KA_modLJ(cell, delta)
    integrator.forces[0] = lj

    sim.run(delta_steps)
    print(delta, thermodynamic_properties.pressure)