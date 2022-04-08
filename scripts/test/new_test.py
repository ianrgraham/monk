import freud
import hoomd
import numpy as np
import gsd.hoomd

BUFFER=0.1

cpu = hoomd.device.CPU()  # or GPU

sim = hoomd.Simulation(device=cpu)

N = 1
phi = 1.2

# Helper functions to setup the snapshot, produces valid states that worked fine
# in v3-beta9
# box, pos = freud.data.UnitCell.sc().generate_system(int(512 ** (1 / 3)))
box = [10, 10, 10, 0, 0, 0]
pos = np.array([[1, 1, 1], [4, 4, 4]], dtype=np.float32)
snap = hoomd.Snapshot()
snap.configuration.box = box
N_actual = len(pos)
snap.particles.N = N_actual
snap.particles.types = ["A"]
snap.particles.typeid[:] = np.zeros(N_actual)
snap.particles.position[:] = pos[:1]

sim.create_state_from_snapshot(snap)

integrator = hoomd.md.Integrator(dt=1e-3)
cell = hoomd.md.nlist.Cell(BUFFER)
pot_pair = hoomd.md.pair.LJ(cell, default_r_cut=2.5)
pot_pair.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
integrator.forces.append(pot_pair)

nvt = hoomd.md.methods.NVT(
    kT=1.5,
    filter=hoomd.filter.All(),
    tau=0.5)
integrator.methods.append(nvt)
sim.operations.integrator = integrator

sim.run(0)