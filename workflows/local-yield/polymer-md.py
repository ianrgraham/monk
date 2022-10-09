import hoomd
import gsd.hoomd
import numpy as np

snap = gsd.hoomd.Snapshot()

# build the snapshot, we are going to build just one molecule
# and then use replicate to make a bunch of them
snap.particles.N = 10
# these can be whatever reasonable values you want
pos = np.zeros((10, 3))
pos[:,:] = -4.5
pos[:,0] += np.linspace(0.0, 5.0/np.sqrt(2.0), 10)
pos[:,1] += np.linspace(0.0, 5.0/np.sqrt(2.0), 10)
pos[:,2] += np.linspace(0.0, 5.0/np.sqrt(2.0), 10)
snap.particles.position = pos

# just big enough to hold the molecule
snap.configuration.dimensions = 3
snap.configuration.box = [10.0, 10.0, 10.0, 0.0, 0.0, 0.0]
snap.particles.types = ['A']
snap.particles.typeid = np.array([0] * 10)
snap.particles.diameter = np.array([1.0] * 10)  # not necessary, but good for visualization

# define the bonds
snap.bonds.N = 9
snap.bonds.group = [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,8], [8,9]]
snap.bonds.types = ['A']

snap.validate()  # will error if something is wrong with the snapshot

device = hoomd.device.GPU()  # hoomd.device.CPU()
sim = hoomd.Simulation(device)

sim.create_state_from_snapshot(snap)
sim.state.replicate(10, 10, 10)  # makes many molecules! 10,000 particles in total

# write initial configuration
hoomd.write.GSD.write(state=sim.state, filename='init.gsd', mode='wb')

# set up integrator, forces and methods
integrator = hoomd.md.Integrator(dt=0.001)

# LJ pair potential
nlist = hoomd.md.nlist.Cell(buffer=0.3)  # or use Tree(), also fiddle with buffer, it may speed up things!
lj = hoomd.md.pair.LJ(nlist, default_r_cut=2.5)
lj.params[('A', 'A')] = {'sigma': 1.0, 'epsilon': 1.0}
integrator.forces.append(lj)

# FENEWCA bond potential
fenewca = hoomd.md.bond.FENEWCA()
fenewca.params['A'] = dict(k=3.0, r0=2.38, epsilon=1.0, sigma=1.0, delta=0.0)
integrator.forces.append(fenewca)

# set up NPT integration method
npt = hoomd.md.methods.NPT(filter=hoomd.filter.All(), kT=1.0, tau=0.1, S=1.0, tauS=1.0, couple='xyz')
integrator.methods.append(npt)
sim.operations.integrator = integrator

# write every 1000 steps
writer = hoomd.write.GSD(filename='equil.gsd', trigger=hoomd.trigger.Periodic(1000), mode='wb')
sim.operations.writers.append(writer)

# eqilibrate
sim.run(100_000)

sim.operations.writers.clear()

variant = hoomd.variant.Ramp(1.0, 0.3, sim.timestep, 100_000)
npt.kT = variant
writer = hoomd.write.GSD(filename='quench.gsd', trigger=hoomd.trigger.Periodic(1000), mode='wb')
sim.operations.writers.append(writer)

sim.run(100_000)