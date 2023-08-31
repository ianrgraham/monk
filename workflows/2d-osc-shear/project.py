#!python

from typing import Tuple
import flow
import signac
import hoomd
import gsd.hoomd
import freud
import numpy as np
import pandas as pd

import glob
import os
import pickle
import copy
import gc
import shutil
import multiprocessing
import pathlib
import math
import itertools

from dataclasses import dataclass
from collections import defaultdict

from monk import workflow, methods, pair, utils, prep
import schmeud._schmeud as _schmeud
from schmeud import ml

config = workflow.get_config()

class Project(flow.FlowProject):
    pass

def _setup_nvt_sim(job: signac.Project.Job, sim: hoomd.Simulation, temp=None):
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc
    dt = proj_doc["dt"]

    pot = sp["pot"]
    if isinstance(pot, str):
        pot = [pot]
    elif isinstance(pot, list):
        pass
    else:
        raise ValueError("The 'potential' must be a string or list")

    integrator = hoomd.md.Integrator(dt=dt)
    pair_func, args = prep.search_for_pair(pot)
    tree = hoomd.md.nlist.Tree(0.3)
    print("Using pair function: ", pair_func)
    force = pair_func(tree, *args)

    if temp is None:
        temp = doc["init_temp"]
    nvt = hoomd.md.methods.NVT(hoomd.filter.All(), temp, dt*100)
    integrator.forces = [force]
    integrator.methods = [nvt]

    sim.operations.integrator = integrator

    return nvt

def _setup_fire_sim(job: signac.Project.Job, sim: hoomd.Simulation, fire_kwargs=None):
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc
    dt = proj_doc["dt"]

    pot = sp["pot"]
    if isinstance(pot, str):
        pot = [pot]
    elif isinstance(pot, list):
        pass
    else:
        raise ValueError("The 'potential' must be a string or list")
    
    default_fire_kwargs = {"dt": 1e-2, "force_tol": 1e-2, "angmom_tol": 1.0, "energy_tol": 1e-6}
    
    if fire_kwargs is None:
        fire_kwargs = default_fire_kwargs
    else:
        default_fire_kwargs.update(fire_kwargs)
        fire_kwargs = default_fire_kwargs

    fire = hoomd.md.minimize.FIRE(**fire_kwargs)
    pair_func, args = prep.search_for_pair(pot)
    tree = hoomd.md.nlist.Tree(0.3)
    print("Using pair function: ", pair_func)
    force = pair_func(tree, *args)

    nve = hoomd.md.methods.NVE(hoomd.filter.All())
    fire.forces = [force]
    fire.methods = [nve]

    sim.operations.integrator = fire

    return fire, nve

@Project.operation
@Project.post.true("initialized")
def init(job: signac.Project.Job):
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc

    step_unit = proj_doc["step_unit"]
    equil_time = proj_doc["equil_time"]
    equil_steps = int(equil_time * step_unit)
    min_periods = proj_doc["min_periods"]
    dumps = proj_doc["dumps"]

    # initialize first simualtion
    if not job.isfile("init.gsd"):
        if sp["prep"] == "HTL":
            print("Initializing simulation")
            sim = prep.quick_sim(sp["N"], sp["rho"], hoomd.device.GPU(), dim=2,
                                ratios=[60, 40], diams=[1.0, 0.88], seed=doc["seed"])
            nvt = _setup_nvt_sim(job, sim)

            sim.run(0)
            sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=doc["init_temp"])
            print("Running initial equilibration")
            sim.run(equil_steps)

            hoomd.write.GSD.write(sim.state, job.fn("init.gsd"))

            del sim, nvt
            gc.collect()
        elif sp["prep"] == "ESL":
            assert job.isfile("start-file.gsd")
            print("Initializing simulation")
            sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=doc["seed"])
            sim.create_state_from_gsd(job.fn("start-file.gsd"))
            nvt = _setup_nvt_sim(job, sim)

            sim.run(0)
            sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=doc["init_temp"])
            print("Running initial equilibration")
            sim.run(equil_steps)

            hoomd.write.GSD.write(sim.state, job.fn("init.gsd"))

            del sim, nvt
            gc.collect()

        else:
            raise ValueError("Other preparation methods are not yet implemented")
    else:
        print("init.gsd found, skipping initialization")

    doc["initialized"] = True

@Project.operation
@Project.pre.true("initialized")
@Project.post.true("experiment_completed")
def shear_experiment(job: signac.Project.Job):
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc

    dt = proj_doc["dt"]
    step_unit = proj_doc["step_unit"]
    equil_time = proj_doc["equil_time"]
    equil_steps = int(equil_time * step_unit)
    min_periods = proj_doc["min_periods"]
    dumps = proj_doc["dumps"]
    period_times = proj_doc["period_times"]
    max_shears = proj_doc["max_shears"]

    

    for temp, period, max_shear in itertools.product(doc["temps"], period_times, max_shears):

        path = pathlib.Path(job.fn(f"experiments/max-shear-{max_shear:.2f}/temp-{temp:.4f}"))
        os.makedirs(path.as_posix(), exist_ok=True)

        outfile = path / f"traj_period-{period}.gsd"
        preequiled_outfile = path / f"preequiled-traj_period-{period}.gsd"  # we might preequil in the future, but not now
        badfile = path / f"traj_period-{period}.gsd.bad"
        print("Current file:", outfile.as_posix())

        if badfile.exists():
            print("Bad file exists, skipping")
            continue

        if outfile.exists():
            file = gsd.hoomd.open(str(outfile), "rb")
            if len(file) >= min_periods * dumps:

                print("File already exists, skipping")
                continue

        if preequiled_outfile.exists():
            file = gsd.hoomd.open(str(preequiled_outfile), "rb")
            if len(file) >= min_periods * dumps:

                print("Preequiled file already exists, skipping")
                continue

        period_steps = int(period * step_unit)
        total_steps = int(min_periods * period_steps)
        output_period = int(period_steps / dumps)

        # make sure period_steps is a multiple of dumps
        assert period_steps % dumps == 0

        try:
            sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=doc["seed"])
            sim.create_state_from_gsd(job.fn("init.gsd"))
            if temp == 0.0:
                fire, nve = _setup_fire_sim(job, sim)
                raise NotImplementedError("Fire is not yet implemented")
                while not fire.converged:
                    sim.run(1000)
            else:
                nvt = _setup_nvt_sim(job, sim, temp=temp)

                sim.run(0)
                sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=temp)

            # don't run preruns
            sim.run(equil_steps)

            thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
            logger = hoomd.logging.Logger()
            logger.add(thermo, quantities=["pressure", "pressure_tensor", "kinetic_temperature"])
            sim.operations.computes.clear()
            sim.operations.computes.append(thermo)

            gsd_writer = hoomd.write.GSD(hoomd.trigger.Periodic(output_period), outfile, mode="wb", log=logger)
            sim.operations.writers.clear()
            sim.operations.writers.append(gsd_writer)

            sim.operations.updaters.clear()
            trigger = hoomd.trigger.Periodic(1)
            variant = methods.Oscillate(1, sim.timestep, period_steps)
            old_box = sim.state.box
            new_box = copy.deepcopy(old_box)
            old_box.xy = -max_shear
            new_box.xy = max_shear
            updater = hoomd.update.BoxResize(trigger, old_box, new_box, variant)
            sim.operations.updaters.append(updater)

            sim.run(total_steps)
            
        except KeyboardInterrupt:
            raise
        except:
            print("Error during run, saving bad file")
            shutil.move(outfile.as_posix(), badfile.as_posix())
        finally:
            try:
                logger.remove(thermo)
                sim.operations.computes.clear()
                del thermo
                print("Finished run")
            except NameError:
                print("Something weird happened, but run is over")
            

    doc["experiment_completed"] = True


@dataclass(frozen=True, eq=True)
class Statepoint:
    max_shear: float
    period: float
    temp: float
    prep: str


@Project.operation
@Project.pre.true("experiment_completed")
@Project.post.true("t1_counts_completed")
def t1_counts(job: signac.Project.Job):
    doc = job.doc
    experiments = sorted(glob.glob(job.fn("experiments/*/*/traj_*.gsd")))
    # prep = job.sp["prep"]

    for exper in experiments:
        max_shear = utils.extract_between(exper, "max-shear-", "/")
        period = utils.extract_between(exper, "period-", ".gsd")
        temp = utils.extract_between(exper, "temp-", "/")
        output_file = job.fn(f"experiments/max-shear-{max_shear}/temp-{temp}/t1-counts_period-{period}.parquet")
        if os.path.exists(output_file):
            continue
        # sp = Statepoint(max_shear=float(max_shear), period=float(period), temp=float(temp), prep=prep)
        
        if float(period) != 1000.0:
            continue

        traj = gsd.hoomd.open(exper)

        print(max_shear, period, temp)

        rev_count = []
        irr_count = []
        voro = freud.locality.Voronoi()
        for i in range(1, 20):

            rearranged = set()

            snap_0 = traj[-1 + i*40] # initial state
            snap_1 = traj[9 + i*40] # peak
            snap_2 = traj[19 + i*40] # back to zero
            snap_3 = traj[29 + i*40] # min peak
            snap_4 = traj[-1 + (i+1)*40] # full cycle complete

            box_0 = snap_0.configuration.box[:]
            box_1 = snap_1.configuration.box[:]
            box_2 = snap_2.configuration.box[:]
            box_3 = snap_3.configuration.box[:]
            box_4 = snap_4.configuration.box[:]

            voro.compute((box_0, snap_0.particles.position))
            nlist = voro.nlist
            neighbors = set([frozenset(set([i, j])) for i, j in zip(nlist.query_point_indices, nlist.point_indices)])

            next_to_process = zip([box_1, box_2, box_3, box_4], [snap_1, snap_2, snap_3, snap_4])

            for box, snap in next_to_process:
                voro.compute((box, snap.particles.position))
                nlist = voro.nlist
                neighbors_ = set([frozenset(set([i, j])) for i, j in zip(nlist.query_point_indices, nlist.point_indices)])
                rearranged |= neighbors - neighbors_
            rev = rearranged & neighbors_
            irr = rearranged - rev
            rev_count.append(len(rev))
            irr_count.append(len(irr))
            # break
        dataset = pd.DataFrame({"rev": rev_count, "irr": irr_count})
        dataset.to_parquet(output_file)

    doc["t1_counts_completed"] = True

if __name__ == "__main__":
    Project.get_project(root=config["root"]).main()