import hoomd

mc = hoomd.hpmc.integrate.ConvexPolyhedron()
mc.shape['octahedron'] = dict(vertices=[
    (-0.5, 0, 0),
    (0.5, 0, 0),
    (0, -0.5, 0),
    (0, 0.5, 0),
    (0, 0, -0.5),
    (0, 0, 0.5),
])

cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu, seed=20)
sim.operations.integrator = mc
# See HOOMD tutorial for how to construct an initial configuration 'init.gsd'
sim.create_state_from_gsd(filename='init.gsd')

sim.run(1e4)
