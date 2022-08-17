import os
import sys
import tempfile

from pathlib import Path

import hoomd
import numpy as np

from monk import pair, prep

out_dir = Path(os.environ["MONK_DATA_DIR"]) / "monk/test"
os.makedirs(out_dir, exist_ok=True)

# use SLURM_ARRAY_ID to define temperature
sim_temp = 0.47
init_temp = 1.5
# the array_id is used as the random seed for numpy to place particles in the
# box and to start the initial random velocities of the nvt simulation

N = 512
phi = 1.2

# initialize hoomd state
print("Init context")
device = hoomd.device.GPU()
sim = hoomd.Simulation(device=device, seed=2)

print(N)

# create equilibrated 3D LJ configuration
rng = prep.init_rng(0)  # random generator for placing particles
snapshot = prep.approx_euclidean_snapshot(N,
                                          np.cbrt(N / phi),
                                          rng,
                                          dim=3,
                                          particle_types=['A', 'B'],
                                          ratios=[80, 20])
sim.create_state_from_snapshot(snapshot)

# set simtential
print("Set LJ potential")
integrator = hoomd.md.Integrator(dt=0.0025)
cell = hoomd.md.nlist.Cell()
lj = pair.LJ(cell)
integrator.forces.append(lj)
variant = hoomd.variant.Ramp(init_temp, sim_temp, int(4e3), int(8e3))
nvt = hoomd.md.methods.NVT(kT=variant, filter=hoomd.filter.All(), tau=0.5)
integrator.methods.append(nvt)

sim.operations.integrator = integrator

sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=init_temp)

thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamic_properties)

logger = hoomd.logging.Logger()

logger.add(thermodynamic_properties)
logger.add(sim, quantities=['timestep', 'walltime'])

with tempfile.TemporaryDirectory() as tmpdir:
    tempdir = "."
    path = 'md_test.gsd'
    print(path)
    gsd_writer = hoomd.write.GSD(filename=str(path),
                                 trigger=hoomd.trigger.Periodic(200),
                                 mode='wb',
                                 filter=hoomd.filter.All())
    sim.operations.writers.append(gsd_writer)
    gsd_writer.log = logger

    sim.run(0)
    print("start:", thermodynamic_properties.pressure,
          thermodynamic_properties.volume)

    # run initial thermalization
    sim.run(4e3)  # 1_000
    print("therm:", thermodynamic_properties.pressure,
          thermodynamic_properties.volume)

    #
    print(integrator.forces[0])
    integrator.forces.clear()
    lj = pair.LJ1208(cell)
    integrator.forces.append(lj)
    print(integrator.forces[0])

    for i in range(10):
        sim.run(4e2)  # 1_000 second
        print("quench:", thermodynamic_properties.pressure,
              thermodynamic_properties.volume)

    for i in range(100):
        sim.run(4e2)  # 10_000 seconds
        print("equil:", thermodynamic_properties.pressure,
              thermodynamic_properties.volume)
        pressure = thermodynamic_properties.pressure

    npt = hoomd.md.methods.NPT(kT=variant,
                               S=pressure,
                               filter=hoomd.filter.All(),
                               tau=0.5,
                               tauS=0.5,
                               couple="none")

    print(integrator.methods[0])
    integrator.methods.clear()
    integrator.methods.append(npt)
    print(integrator.methods[0])

    for i in range(100):
        sim.run(4e2)  # 10_000 seconds
        print("cp:", thermodynamic_properties.pressure,
              thermodynamic_properties.volume)
