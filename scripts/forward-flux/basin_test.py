import pathlib
import tempfile
import argparse
import sys

from inspect import signature

import hoomd
import numpy as np

from monk import prep, pair

valid_output_formats = [".gsd"]

parser = argparse.ArgumentParser(description="Initialize a system of \
        particles (preferably in the liquid state), equilibrate the configuration, \
        and then quench to a lower temperature (in the glassy or supercooled regime). \
        From there we continue to run the simulation.")
parser.add_argument("ofile", type=str, help=f"Output file (allowed formats: {valid_output_formats}")
parser.add_argument("--num", type=int, help="Number of particles to simulate.", default=256)
parser.add_argument("--pair", nargs="+", help="Set the potential pair with any function callable in 'monk.pair'", default=["KA_LJ"])
parser.add_argument("--dt", type=float, default=2.5e-3, help="Timestep size")
parser.add_argument("--phi", type=float, default=1.2, help="Volume fraction")
parser.add_argument("--temps", type=float, nargs=2, default=[1.5, 0.47], help="Starting and ending temperatures")
parser.add_argument("--equil-time", type=int, default=1e2, help="Time to equilibrate before quenching")
parser.add_argument("--quench-rate", type=float, default=1e-2, help="Rate (dT/dt) at which to change the temperature")
parser.add_argument("--dump-rate", type=float, default=1, help="Rate at which to write to disk")
parser.add_argument("--throw-away", type=int, default=1e2, help="Additional time to wait before writing")
parser.add_argument("--sim-time", type=int, default=1e3, help="Total time to simulate post-quench")
parser.add_argument("--seed", type=int, help="Random seed to initialize the RNG.", default=27)
parser.add_argument("--dump-setup", action="store_true", help="Start recording data right away")
parser.add_argument("--scratch", action="store_true", help="Use scratch space ")

args = parser.parse_args()

ofile = pathlib.Path(args.ofile)

N = args.num
(init_temp, sim_temp) = tuple(args.temps)
dT = sim_temp - init_temp
dt = args.dt
phi = args.phi
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
device = hoomd.device.auto_select()
sim = hoomd.Simulation(device=device, seed=seed)
print(f"Running on {device.devices[0]}")

rng = prep.init_rng(seed)  # random generator for placing particles
snapshot = prep.approx_euclidean_snapshot(
    N,
    np.cbrt(N / phi),
    rng,
    dim=3,
    particle_types=['A', 'B'],
    ratios=[80, 20])

typeids = snapshot.particles.typeid

lj_diam = lambda _type: 1.0 if _type == 0 else 0.88

snapshot.particles.diameter = [lj_diam(t) for t in typeids]
sim.create_state_from_snapshot(snapshot)


# set simtential
print(f"Set potential. {{ pair: {pair_name}, args: {arguments} }}")
integrator = hoomd.md.Integrator(dt=dt)
cell = hoomd.md.nlist.Cell()
pot_pair = pair_func(cell, *arguments)
integrator.forces.append(pot_pair)

# start and end of ramp
start = int(args.equil_time / dt)
end = start + int(abs(dT / dt / args.quench_rate))

variant = hoomd.variant.Ramp(init_temp, sim_temp, int(start), int(end))
nvt = hoomd.md.methods.NVT(
    kT=variant,
    filter=hoomd.filter.All(),
    tau=0.5)
integrator.methods.append(nvt)
sim.operations.integrator = integrator

sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=init_temp)

proto_sim_steps = int(args.sim_time / dt)

print("Running setup/equilibration")
if args.dump_setup:
    sim.run(0)
    sim_steps = end + throw_away + proto_sim_steps
else:
    sim.run(end + throw_away)
    sim_steps = proto_sim_steps

print("Applying fire minimization before data is collected")
fire_integrator = hoomd.md.minimize.FIRE(dt=dt)
integrator.forces.clear()
fire_integrator.forces.append(pot_pair)
nvt = hoomd.md.methods.NVE(filter=hoomd.filter.All())
fire_integrator.methods.append(nvt)
sim.operations.integrator = fire_integrator

idx = 0
while not fire_integrator.converged:
    print(f"Fire run {idx}")
    sim.run(1000)
    idx += 1

# Back to the normal integrator
fire_integrator.forces.clear()
integrator.forces.append(pot_pair)
sim.operations.integrator = integrator

sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=init_temp)

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
                            log = logger)

sim.operations.writers.append(gsd_writer)
gsd_writer.log = logger

print("Running simulation")
sim.run(sim_steps)
