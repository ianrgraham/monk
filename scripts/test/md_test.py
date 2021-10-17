# From the tutorial on MD in HOOMD v3

import itertools
import math
import tempfile

from pathlib import Path

import gsd.hoomd
import hoomd
import numpy

m = 4
N_particles = 4 * m**3

spacing = 1.3
K = math.ceil(N_particles**(1 / 3))
L = K * spacing
x = numpy.linspace(-L / 2, L / 2, K, endpoint=False)
position = list(itertools.product(x, repeat=3))

snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = N_particles
snapshot.particles.position = position[0:N_particles]
snapshot.particles.typeid = [0] * N_particles
snapshot.configuration.box = [L, L, L, 0, 0, 0]

snapshot.particles.types = ['A']

gpu = hoomd.device.GPU()
sim = hoomd.Simulation(device=gpu, seed=1)
sim.create_state_from_snapshot(snapshot)

integrator = hoomd.md.Integrator(dt=0.005)
cell = hoomd.md.nlist.Cell()
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
lj.r_cut[('A', 'A')] = 2.5
integrator.forces.append(lj)
nvt = hoomd.md.methods.NVT(kT=1.5, filter=hoomd.filter.All(), tau=1.0)
integrator.methods.append(nvt)

sim.operations.integrator = integrator

sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)

thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamic_properties)
sim.run(1000)

logger = hoomd.logging.Logger()

logger.add(thermodynamic_properties)
logger.add(sim, quantities=['timestep', 'walltime'])

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / 'md_test.gsd'
    print(path)
    gsd_writer = hoomd.write.GSD(filename=str(path),
                                trigger=hoomd.trigger.Periodic(200),
                                mode='xb',
                                filter=hoomd.filter.All())
    sim.operations.writers.append(gsd_writer)
    gsd_writer.log = logger
    for i in range(10):
        sim.run(10000)
        print(thermodynamic_properties.pressure)

    # end simulation
    del sim, gsd_writer, thermodynamic_properties, logger
    del integrator, nvt, lj, cell, gpu

    traj = gsd.hoomd.open(path, 'rb')

    print("Log output:\n", traj[0].log)
