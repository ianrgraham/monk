import itertools
import math

import hoomd
import numpy

from hoomd.simulation import Simulation

def print_sim_info(sim: Simulation):
    print(sim.state)
    (sim.operations.integrator)
    print(sim.operations.updaters[:])
    print(sim.operations.writers[:])

gpu = hoomd.device.GPU()

sim = hoomd.Simulation(device=gpu)
print_sim_info(sim)

mc = hoomd.hpmc.integrate.ConvexPolyhedron()
mc.shape['octahedron'] = dict(vertices=[
    (-0.5, 0, 0),
    (0.5, 0, 0),
    (0, -0.5, 0),
    (0, 0.5, 0),
    (0, 0, -0.5),
    (0, 0, 0.5),
])

mc.nselect = 2
mc.d['octahedron'] = 0.15
mc.a['octahedron'] = 0.2

sim.operations.integrator = mc

m = 4
N_particles = 2 * m**3

spacing = 1.2
K = math.ceil(N_particles**(1 / 3))
L = K * spacing

x = numpy.linspace(-L / 2, L / 2, K, endpoint=False)
position = list(itertools.product(x, repeat=3))
print(position[0:4])

position = position[0:N_particles]

orientation = [(1, 0, 0, 0)] * N_particles

render(position, orientation, L)

