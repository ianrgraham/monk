#!python

from typing import Tuple
import flow
import hoomd.update.box_resize
import signac
import hoomd
import gsd.hoomd
import freud
import numpy as np
import pandas as pd
import polars as pl

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

from tqdm import tqdm

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

def _setup_fire_sim(job: signac.Project.Job, sim: hoomd.Simulation, fire_kwargs=None, method=None):
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

    if method is None:
        nve = hoomd.md.methods.NVE(hoomd.filter.All())
    else:
        nve = method
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
    # dumps = proj_doc["dumps"]
    dumps = 4
    period_times = proj_doc["period_times"]
    # max_shears = proj_doc["max_shears"]
    max_shears = [0.05]

    

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



@Project.operation
@Project.pre.true("initialized")
@Project.post.true("init_fire_applied")
def init_fire_minimize(job: signac.Project.Job):
    sp = job.sp
    doc = job.doc
    print(job)

    sim = hoomd.Simulation(hoomd.device.GPU(), seed=doc["seed"])
    sim.create_state_from_gsd(job.fn("init.gsd"))
    fire, nve = _setup_fire_sim(job, sim)

    # setup details for sims
    runs = job.fn("init.gsd")
    for run in [runs]:
        print(run)
        traj = gsd.hoomd.open(run)
        orig_len = len(traj)
        print(orig_len)
        fire_traj = run.replace("init", "init-fire")
        print(run)
        print(fire_traj)

        assert fire_traj != traj

        if os.path.isfile(fire_traj):
            tmp_traj = gsd.hoomd.open(fire_traj)
            len_traj = len(tmp_traj)
            del tmp_traj
            if len_traj == orig_len:
                print("run is done")
                continue
            else:
                print("rerunning")

        methods.fire_minimize_frames(
            sim,
            traj,
            fire_traj,
            fire_steps=100
        )
        print("done")
    print("all done!")
    job.doc["init_fire_applied"] = True


@Project.operation
@Project.pre.after(shear_experiment)
@Project.post.true("fire_applied")
def fire_minimize(job: signac.Project.Job):
    sp = job.sp
    doc = job.doc
    print(job)

    sim = hoomd.Simulation(hoomd.device.GPU(), seed=doc["seed"])
    sim.create_state_from_gsd(job.fn("init.gsd"))
    fire, nve = _setup_fire_sim(job, sim)

    # setup details for sims
    runs = glob.glob(job.fn(f"experiments/max-shear-*/temp-*/traj_period-1000.0.gsd"))
    for run in runs:
        print(run)
        traj = gsd.hoomd.open(run)
        orig_len = len(traj)
        print(orig_len)
        fire_traj = run.replace("traj_period", "traj-fire_period")
        print(run)
        print(fire_traj)

        assert fire_traj != traj

        if os.path.isfile(fire_traj):
            tmp_traj = gsd.hoomd.open(fire_traj)
            len_traj = len(tmp_traj)
            del tmp_traj
            if len_traj == orig_len:
                print("run is done")
                continue
            else:
                print("rerunning")

        methods.fire_minimize_frames(
            sim,
            traj,
            fire_traj,
            fire_steps=100
        )
        print("done")
    print("all done!")
    job.doc["fire_applied"] = True


@Project.operation
@Project.pre.true("initialized")
@Project.post.true("experiment_other_completed")
def shear_other_experiment(job: signac.Project.Job):
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc

    dt = proj_doc["dt"]
    step_unit = proj_doc["step_unit"]
    equil_time = proj_doc["equil_time"]
    equil_steps = int(equil_time * step_unit)
    min_periods = 5
    # dumps = proj_doc["dumps"]
    dumps = 4
    period_times = proj_doc["period_times"]
    # max_shears = proj_doc["max_shears"]
    max_shears = [0.05]

    

    for temp, period, max_shear in itertools.product(doc["temps"], period_times, max_shears):

        path = pathlib.Path(job.fn(f"other_experiments/max-shear-{max_shear:.2f}/temp-{temp:.4f}"))
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

            gsd_writer = hoomd.write.GSD(hoomd.trigger.Periodic(output_period, phase=sim.timestep), outfile, mode="wb", log=logger)
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
            

    doc["experiment_other_completed"] = True


@Project.operation
@Project.pre.true("experiment_completed")
@Project.post.true("continue_experiment_completed")
def continue_shear_experiment(job: signac.Project.Job):
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc

    dt = proj_doc["dt"]
    step_unit = proj_doc["step_unit"]
    equil_time = proj_doc["equil_time"]
    equil_steps = int(equil_time * step_unit)
    min_periods = 5.25
    # dumps = proj_doc["dumps"]
    dumps = 4
    period_times = proj_doc["period_times"]
    # max_shears = proj_doc["max_shears"]
    max_shears = [0.05]

    

    for temp, period, max_shear in itertools.product(doc["temps"], period_times, max_shears):

        path = pathlib.Path(job.fn(f"continue_experiments/max-shear-{max_shear:.2f}/temp-{temp:.4f}"))
        previous = pathlib.Path(path.as_posix().replace("continue_experiments", "experiments"))
        if not previous.exists():
            print("No previous file found")
            continue

        os.makedirs(path.as_posix(), exist_ok=True)

        load_file = previous / f"traj_period-{period}.gsd"
        outfile = path / f"traj_period-{period}.gsd"
        preequiled_outfile = path / f"preequiled-traj_period-{period}.gsd"  # we might preequil in the future, but not now
        badfile = path / f"traj_period-{period}.gsd.bad"
        print("Current file:", outfile.as_posix())

        # if badfile.exists():
        #     print("Bad file exists, skipping")
        #     continue

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
            init = gsd.hoomd.open(job.fn("init.gsd"))
            init_timestep = init[-1].configuration.step
            sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=doc["seed"])
            sim.create_state_from_gsd(load_file.as_posix())
            if temp == 0.0:
                fire, nve = _setup_fire_sim(job, sim)
                raise NotImplementedError("Fire is not yet implemented")
                while not fire.converged:
                    sim.run(1000)
            else:
                nvt = _setup_nvt_sim(job, sim, temp=temp)

                sim.run(0)
                # sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=temp)

            # don't run preruns
            # sim.run(equil_steps)


            thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
            logger = hoomd.logging.Logger()
            logger.add(thermo, quantities=["pressure", "pressure_tensor", "kinetic_temperature"])
            sim.operations.computes.clear()
            sim.operations.computes.append(thermo)

            gsd_writer = hoomd.write.GSD(hoomd.trigger.Periodic(output_period, phase=int(init_timestep + equil_steps)), outfile, mode="wb", log=logger)
            sim.operations.writers.clear()
            sim.operations.writers.append(gsd_writer)

            sim.operations.updaters.clear()
            trigger = hoomd.trigger.Periodic(1)
            variant = methods.Oscillate(1, int(init_timestep + equil_steps), period_steps)
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
            

    doc["continue_experiment_completed"] = True

@Project.operation
@Project.pre.true("experiment_completed")
@Project.post.true("continue_longer_experiment_completed")
def continue_longer_shear_experiment(job: signac.Project.Job):
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc

    dt = proj_doc["dt"]
    step_unit = proj_doc["step_unit"]
    equil_time = proj_doc["equil_time"]
    equil_steps = int(equil_time * step_unit)
    min_periods = 100.25
    # dumps = proj_doc["dumps"]
    dumps = 4
    period_times = proj_doc["period_times"]
    # max_shears = proj_doc["max_shears"]
    max_shears = [0.05]

    

    for temp, period, max_shear in itertools.product(doc["temps"], period_times, max_shears):

        path = pathlib.Path(job.fn(f"continue_longer_experiments/max-shear-{max_shear:.2f}/temp-{temp:.4f}"))
        previous = pathlib.Path(path.as_posix().replace("continue_longer_experiments", "experiments"))
        if not previous.exists():
            print("No previous file found")
            continue

        os.makedirs(path.as_posix(), exist_ok=True)

        load_file = previous / f"traj_period-{period}.gsd"
        outfile = path / f"traj_period-{period}.gsd"
        preequiled_outfile = path / f"preequiled-traj_period-{period}.gsd"  # we might preequil in the future, but not now
        badfile = path / f"traj_period-{period}.gsd.bad"
        print("Current file:", outfile.as_posix())

        # if badfile.exists():
        #     print("Bad file exists, skipping")
        #     continue

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
            init = gsd.hoomd.open(job.fn("init.gsd"))
            init_timestep = init[-1].configuration.step
            sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=doc["seed"])
            sim.create_state_from_gsd(load_file.as_posix())
            if temp == 0.0:
                fire, nve = _setup_fire_sim(job, sim)
                raise NotImplementedError("Fire is not yet implemented")
                while not fire.converged:
                    sim.run(1000)
            else:
                nvt = _setup_nvt_sim(job, sim, temp=temp)

                sim.run(0)
                # sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=temp)

            # don't run preruns
            # sim.run(equil_steps)


            thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
            logger = hoomd.logging.Logger()
            logger.add(thermo, quantities=["pressure", "pressure_tensor", "kinetic_temperature"])
            sim.operations.computes.clear()
            sim.operations.computes.append(thermo)

            gsd_writer = hoomd.write.GSD(hoomd.trigger.Periodic(output_period, phase=int(init_timestep + equil_steps)), outfile, mode="wb", log=logger)
            sim.operations.writers.clear()
            sim.operations.writers.append(gsd_writer)

            sim.operations.updaters.clear()
            trigger = hoomd.trigger.Periodic(1)
            variant = methods.Oscillate(1, int(init_timestep + equil_steps), period_steps)
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
            

    doc["continue_longer_experiment_completed"] = True

@Project.operation
@Project.pre.after(continue_shear_experiment)
@Project.post.true("continue_fire_applied")
def continue_fire_minimize(job: signac.Project.Job):
    sp = job.sp
    doc = job.doc
    print(job)

    sim = hoomd.Simulation(hoomd.device.GPU(), seed=doc["seed"])
    sim.create_state_from_gsd(job.fn("init.gsd"))
    fire, nve = _setup_fire_sim(job, sim)

    # setup details for sims
    runs = glob.glob(job.fn(f"continue_experiments/max-shear-*/temp-*/traj_period-1000.0.gsd"))
    for run in runs:
        print(run)
        traj = gsd.hoomd.open(run)
        orig_len = len(traj)
        print(orig_len)
        fire_traj = run.replace("traj_period", "traj-fire_period")
        print(run)
        print(fire_traj)

        assert fire_traj != traj

        if os.path.isfile(fire_traj):
            tmp_traj = gsd.hoomd.open(fire_traj)
            len_traj = len(tmp_traj)
            del tmp_traj
            if len_traj == orig_len:
                print("run is done")
                continue
            else:
                print("rerunning")

        methods.fire_minimize_frames(
            sim,
            traj,
            fire_traj,
            fire_steps=100
        )
        print("done")
    print("all done!")
    job.doc["continue_fire_applied"] = True

@Project.operation
@Project.pre.after(continue_longer_shear_experiment)
@Project.post.true("continue_longer_fire_applied")
def continue_longer_fire_minimize(job: signac.Project.Job):
    sp = job.sp
    doc = job.doc
    print(job)

    sim = hoomd.Simulation(hoomd.device.GPU(), seed=doc["seed"])
    sim.create_state_from_gsd(job.fn("init.gsd"))
    fire, nve = _setup_fire_sim(job, sim)

    # setup details for sims
    runs = glob.glob(job.fn(f"continue_longer_experiments/max-shear-*/temp-*/traj_period-1000.0.gsd"))
    for run in runs:
        print(run)
        traj = gsd.hoomd.open(run)
        orig_len = len(traj)
        print(orig_len)
        fire_traj = run.replace("traj_period", "traj-fire_period")
        print(run)
        print(fire_traj)

        assert fire_traj != traj

        if os.path.isfile(fire_traj):
            tmp_traj = gsd.hoomd.open(fire_traj)
            len_traj = len(tmp_traj)
            del tmp_traj
            if len_traj == orig_len:
                print("run is done")
                continue
            else:
                print("rerunning")

        methods.fire_minimize_frames(
            sim,
            traj,
            fire_traj,
            fire_steps=100
        )
        print("done")
    print("all done!")
    job.doc["continue_longer_fire_applied"] = True

# 0.001983645726339585
@Project.operation
@Project.pre.true("initialized")
@Project.post.true("experiment_short_completed")
def short_shear_experiment(job: signac.Project.Job):
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc

    dt = proj_doc["dt"]
    step_unit = proj_doc["step_unit"]
    equil_time = proj_doc["equil_time"]
    equil_steps = int(equil_time * step_unit)
    min_periods = proj_doc["min_periods"]
    # dumps = proj_doc["dumps"]
    dumps = 4
    period_times = proj_doc["period_times"]
    # max_shears = proj_doc["max_shears"]
    max_shears = [0.05]
    temps = [0.001983645726339585]

    run_temps = [0.3, 0.337, 0.375, 0.412, 0.45]
    if doc["init_temp"] not in run_temps:
        return

    

    for temp, period, max_shear in itertools.product(temps, period_times, max_shears):

        path = pathlib.Path(job.fn(f"short_experiments/max-shear-{max_shear:.2f}/temp-{temp:.4f}"))
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

            gsd_writer = hoomd.write.GSD(hoomd.trigger.Periodic(output_period, phase=sim.timestep), outfile, mode="wb", log=logger)
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

@Project.operation
@Project.pre.true("initialized")
@Project.post.true("aqs_linear_experiment_completed")
def aqs_linear_experiment(job: signac.Project.Job):
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc

    dt = proj_doc["dt"]
    step_unit = proj_doc["step_unit"]
    equil_time = proj_doc["equil_time"]
    equil_steps = int(equil_time * step_unit)
    # min_periods = proj_doc["min_periods"]
    # min_periods = 80
    # dumps = 150 # proj_doc["dumps"]
    max_shears = [0.20] # proj_doc["max_shears"]

    temp = 0.0

    step_sizes = [0.001] # , 0.0001]
    

    for max_shear, step_size in itertools.product(max_shears, step_sizes):


        period = step_size
        path = pathlib.Path(job.fn(f"aqs_linear_experiments/max-shear-{max_shear:.2f}/temp-{temp:.4f}"))
        os.makedirs(path.as_posix(), exist_ok=True)

        outfile = path / f"traj_period-{period}_more-fire.gsd"
        preequiled_outfile = path / f"preequiled-traj_period-{period}_more-fire.gsd"  # we might preequil in the future, but not now
        badfile = path / f"traj_period-{period}_more-fire.gsd.bad"
        print("Current file:", outfile.as_posix())

        # if badfile.exists():
        #     print("Bad file exists, skipping")
        #     continue

        # if outfile.exists():
        #     file = gsd.hoomd.open(str(outfile), "rb")
        #     if len(file) >= min_periods * dumps:

        #         print("File already exists, skipping")
        #         continue

        # if preequiled_outfile.exists():
        #     file = gsd.hoomd.open(str(preequiled_outfile), "rb")
        #     if len(file) >= min_periods * dumps:

        #         print("Preequiled file already exists, skipping")
        #         continue

        # quarter_period = int(np.round(max_shear / step_size))

        # period_steps = int(4 * quarter_period)

        xy = np.linspace(0, max_shear, int(max_shear / step_size) + 1)

        # xy = np.concatenate([xy_seg[1:], xy_seg[:-1][::-1], -xy_seg[1:], -xy_seg[:-1][::-1]])

        # assert len(xy) == period_steps

        # steps_between_dumps = int(period_steps / dumps)

        # do_dumps = np.zeros(period_steps, dtype=bool)
        # do_dumps[steps_between_dumps-1::steps_between_dumps] = True

        # # make sure period_steps is a multiple of dumps
        # assert period_steps % dumps == 0

        try:
            sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=doc["seed"])
            sim.create_state_from_gsd(job.fn("init.gsd"))
            fire, nve = _setup_fire_sim(job, sim, fire_kwargs={"dt":1e-2, "force_tol": 1e-2, "angmom_tol": 1.0, "energy_tol": 1e-10})
            # raise NotImplementedError("Fire is not yet implemented")
            while not fire.converged:
                sim.run(10_000)

            # don't run preruns
            sim.run(equil_steps)

            thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
            logger = hoomd.logging.Logger()
            logger.add(thermo, quantities=["pressure", "pressure_tensor", "kinetic_temperature"])
            sim.operations.computes.clear()
            sim.operations.computes.append(thermo)

            async_write_trig = methods.AsyncTrigger()

            gsd_writer = hoomd.write.GSD(async_write_trig, outfile, mode="wb", log=logger)
            sim.operations.writers.clear()
            sim.operations.writers.append(gsd_writer)

            sim.operations.updaters.clear()
            # trigger = hoomd.trigger.Periodic(1)
            # variant = methods.Oscillate(1, sim.timestep, period_steps)
            # old_box = sim.state.box
            # new_box = copy.deepcopy(old_box)
            # old_box.xy = -max_shear
            # new_box.xy = max_shear
            # updater = hoomd.update.BoxResize(trigger, old_box, new_box, variant)
            # sim.operations.updaters.append(updater)

            new_box = sim.state.box

            # change box
            # sim.state.set_box()

            # sim.run(total_steps)

            pbar = tqdm(total=len(xy))

            # pbar.set_description(f"Iteration {iter+1}/{min_periods}")
            for xy_val in xy:
                # print(snap.configuration.box)
                # print(snap.particles.position[32439])
                # async_trig.activate()
                new_box.xy = xy_val
                hoomd.update.BoxResize.update(sim.state, new_box)

                sim.run(2)
                fire.reset()
                # print("pre-run")
                while not fire.converged:
                    sim.run(10_000)
                # print("post-run")
                # if do_dump:
                async_write_trig.activate()
                sim.run(2)
                pbar.update(1)
            # reset progress bar
            pbar.reset()
                # pbar.refresh()
                    
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
            

    doc["aqs_linear_experiment_completed"] = True


if __name__ == "__main__":
    Project.get_project(root=config["root"]).main()