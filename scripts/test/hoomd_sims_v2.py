import hoomd
from monk import prep, pair

N = 512

L = prep.len_from_phi(N, 1.2)

rng = prep.init_rng(0)
snap = prep.approx_euclidean_snapshot(N,
                                      L,
                                      rng,
                                      dim=3,
                                      particle_types=["A"],
                                      ratios=[100])

cell = hoomd.md.nlist.Cell(0.1)
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
lj.r_cut[('A', 'A')] = 2.5

integrator = hoomd.md.Integrator(dt=0.005)
integrator.forces.append(lj)
nvt = hoomd.md.methods.NVT(kT=1.5, filter=hoomd.filter.All(), tau=1.0)
integrator.methods.append(nvt)

cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu)
sim.operations.integrator = integrator
# See HOOMD tutorial for how to construct an initial configuration 'init.gsd'
sim.create_state_from_snapshot(snap)
sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)

sim.run(0)
