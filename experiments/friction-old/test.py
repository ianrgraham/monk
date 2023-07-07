# %%
import hoomd
import freud
import schmeud
import gsd.hoomd

import matplotlib.pyplot as plt
import numpy as np

import copy

from monk import prep, methods, pair, render, nb

# %%
start_temp = 1e-2
temp = 3e-4
# S = 1.0
# Sxy = 3e-3

dt = 0.01
steps = 90_000
strain_rate = 0.001
max_strain = 1.0
strain = strain_rate * steps * dt
assert strain < max_strain
for i in range(1):
    # %%
    sim = prep.quick_sim(256, 1.4, hoomd.device.CPU(), dim=2, diams=[14/12, 10/12])
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=start_temp)

    # %%
    integrator = hoomd.md.Integrator(dt=dt)

    nlist = hoomd.md.nlist.Cell(0.3)
    hertz = pair.bi_hertz(nlist)
    # drag = methods.SimpleViscousForce(1.0)
    integrator.forces = [hertz]

    nvt = hoomd.md.methods.NVT(hoomd.filter.All(), kT=start_temp, tau=1.0)
    integrator.methods.append(nvt)

    sim.operations.integrator = integrator

    # %%
    sim.run(10_000)


    # %%
    nvt.kT = temp
    sim.run(10_000)

    # %%
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    logger = hoomd.logging.Logger()
    logger.add(thermo, quantities=["pressure", "pressure_tensor", "kinetic_temperature"])
    sim.operations.computes.clear()
    sim.operations.computes.append(thermo)

    gsd_writer = hoomd.write.GSD(hoomd.trigger.Periodic(100), "cs-test.gsd", mode="wb", log=logger)
    sim.operations.writers.clear()
    sim.operations.writers.append(gsd_writer)

    sim.operations.updaters.clear()
    trigger = hoomd.trigger.Periodic(10)
    variant = hoomd.variant.Ramp(0, 1, sim.timestep, steps)
    old_box = sim.state.box
    new_box = copy.deepcopy(old_box)
    new_box.xy = strain
    updater = hoomd.update.BoxResize(trigger, old_box, new_box, variant)
    sim.operations.updaters.append(updater)

    # action = methods.KeepBoxTiltsSmall()
    # updater = hoomd.update.CustomUpdater(trigger, action)
    # sim.operations.updaters.append(updater)


    # %%
    sim.run(steps)


    # %%
    sim.state.box

    # %%
    traj = gsd.hoomd.open("cs-test.gsd", "rb")

    # %%
    xy = [snap.configuration.box[3] for snap in traj]
    plt.plot(xy)

    # %%
    traj[0].log

    # %%
    sigma_xy = [snap.log["md/compute/ThermodynamicQuantities/pressure_tensor"][3] for snap in traj]
    plt.plot(sigma_xy)

    # %%
    kin_temps = [snap.log["md/compute/ThermodynamicQuantities/kinetic_temperature"] for snap in traj]
    plt.plot(kin_temps)

# %%



