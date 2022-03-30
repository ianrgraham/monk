import hoomd
from monk import prep, pair
import time
import numpy as np
import gsd.hoomd

from typing import Optional, List

start = time.time()

cpu = hoomd.device.CPU()

sim = hoomd.Simulation(device=cpu)

N = 512
phi = 1.2

L = prep.len_from_phi(N, phi)

print(L)

rng = prep.init_rng(0)
snap = prep.approx_euclidean_snapshot(N, L, rng, dim=3, particle_types=["A", "B"], ratios=[80, 20])

sim.create_state_from_snapshot(snap)

print("Loaded snap!")

integrator = hoomd.md.Integrator(dt=1e-3)
nlist = hoomd.md.nlist.Tree(0.2)
pot_pair = pair.KA_LJ(nlist)
integrator.forces.append(pot_pair)

print("Made forces!")

# start and end of ramp
nvt = hoomd.md.methods.NVT(
    kT=1.5,
    filter=hoomd.filter.All(),
    tau=0.5)
integrator.methods.append(nvt)
print("Made integrator!")
sim.operations.integrator = integrator

print("Let's load it all up and run!")

sim.run(10_000)

print("Success!")

print(time.time() - start)

# del sim

# print("blah")

# sim2 = hoomd.Simulation(device=cpu)

# sim2.create_state_from_snapshot(snap)

# integrator = hoomd.md.Integrator(dt=0.5)
# cell = hoomd.md.nlist.Cell()
# pot_pair = pair.KA_LJ(cell)
# integrator.forces.append(pot_pair)

# # start and end of ramp
# nvt = hoomd.md.methods.NVT(
#     kT=1.5,
#     filter=hoomd.filter.All(),
#     tau=0.5)
# integrator.methods.append(nvt)
# sim2.operations.integrator = integrator

# sim2.run(10000)