# %%
import hoomd
import freud
import schmeud

import numpy as np

from monk import prep, methods, pair, render, nb

# %%
temp = 1e-3

# %%
sim = prep.quick_sim(256, 0.5, hoomd.device.CPU(), dim=2)
sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=temp)

# %%
integrator = hoomd.md.Integrator(dt=0.01)

nlist = hoomd.md.nlist.Cell(0.3)
hertz = pair.bi_hertz(nlist)
integrator.forces.append(hertz)

nvt = hoomd.md.methods.NPT(hoomd.filter.All(), kT=temp, tau=1.0, S=0.5, tauS=1.2, couple="xy", gamma=0.1)
integrator.methods.append(nvt)

sim.operations.integrator = integrator

gsd_writer = hoomd.write.GSD(hoomd.trigger.Periodic(1000), "test.gsd", mode="wb")
sim.operations.writers.clear()
sim.operations.writers.append(gsd_writer)

action = methods.KeepBoxTiltsSmall()
updater = hoomd.update.CustomUpdater(hoomd.trigger.Periodic(10), action)
sim.operations.updaters.append(updater)

# %%
sim.run(100_000)


