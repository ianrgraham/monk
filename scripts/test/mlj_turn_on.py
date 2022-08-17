import pathlib
import tempfile
import argparse
import sys
import matplotlib.pyplot as plt

from inspect import signature

import hoomd
import gsd
import numpy as np

from monk import prep, pair, plot

valid_output_formats = [".gsd"]

parser = argparse.ArgumentParser(
    description="Initialize a packing of LJ-like"
    " particles, equilibrate them, and then quench the configuration.")
parser.add_argument(
    "ofile",
    type=str,
    help=f"Output file (allowed formats: {valid_output_formats}")
parser.add_argument("--num",
                    type=int,
                    help="Number of particles to simulate.",
                    default=512)
parser.add_argument("--dt", type=float, default=1e-3)
parser.add_argument("--phi", type=float, default=1.2)
parser.add_argument("--temp", type=float, default=1.0)
parser.add_argument("--deltas", type=float, nargs=3, default=[0.0, 0.75, 16])
parser.add_argument("--equil-time", type=int, default=100)
parser.add_argument("--step-time", type=float, default=10)
parser.add_argument("--dump-rate", type=float, default=1)
parser.add_argument("--seed",
                    type=int,
                    help="Random seed to initialize the RNG.",
                    default=27)

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
device = hoomd.device.GPU()
sim = hoomd.Simulation(device=device, seed=seed)
print(f"Running on {device.devices[0]}")

# create equilibrated 3D LJ configuration
rng = prep.init_rng(seed)  # random generator for placing particles
snapshot = prep.approx_euclidean_snapshot(N,
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
pot_pair = pair.KA_ModLJ(cell, 0.0)
integrator.forces.append(pot_pair)

# start and end of ramp
equil_steps = int(args.equil_time / dt)
delta_steps = int(args.step_time / dt)

nvt = hoomd.md.methods.NVT(kT=temp, filter=hoomd.filter.All(), tau=0.5)
integrator.methods.append(nvt)
sim.operations.integrator = integrator

sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=temp)

thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamic_properties)

logger = hoomd.logging.Logger()

logger.add(thermodynamic_properties)
logger.add(sim, quantities=['timestep', 'walltime'])

dump_steps = int(args.dump_rate / dt)

gsd_writer = hoomd.write.GSD(filename=str(ofile),
                             trigger=hoomd.trigger.Periodic(dump_steps),
                             mode='wb',
                             filter=hoomd.filter.All(),
                             log=logger)

sim.operations.writers.append(gsd_writer)

print("Run equilibration")
sim.run(equil_steps)

# del sim, gsd_writer, thermodynamic_properties, logger
# del integrator, nvt, pot_pair, cell, device

traj = gsd.hoomd.open(str(ofile), 'rb')

_, (_, pressures) = plot.scalar_quantity(
    traj, 'md/compute/ThermodynamicQuantities/pressure')

del traj

pressure_A = np.mean(pressures)
pressure_B = np.mean(pressures[-20:])

print(pressure_A, pressure_B)

# plt.show()

print("Swap NVT integrator with NPT")
npt = hoomd.md.methods.NPT(kT=temp,
                           S=pressure_B,
                           filter=hoomd.filter.All(),
                           tau=0.5,
                           tauS=0.5,
                           couple="xyz")

integrator.methods[0] = npt

print("Ramp delta")

for delta in np.linspace(*delta_params):

    lj = pair.KA_ModLJ(cell, delta)
    integrator.forces[0] = lj

    sim.run(delta_steps)
    print(delta, N / thermodynamic_properties.volume)

sim.run(delta_steps * 10)

traj = gsd.hoomd.open(str(ofile), 'rb')

_, (_, pressures) = plot.scalar_quantity(
    traj, 'md/compute/ThermodynamicQuantities/volume')

del traj

plt.show()
