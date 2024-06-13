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

from excess_entropy import *

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


@Project.operation
@Project.pre.true("initialized")
@Project.post.true("aqs_experiment_completed")
def aqs_experiment(job: signac.Project.Job):
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc

    dt = proj_doc["dt"]
    step_unit = proj_doc["step_unit"]
    equil_time = proj_doc["equil_time"]
    equil_steps = int(equil_time * step_unit)
    min_periods = proj_doc["min_periods"]
    # min_periods = 80
    dumps = proj_doc["dumps"]
    max_shears = proj_doc["max_shears"]

    temp = 0.0

    step_sizes = [0.001] # , 0.0001]
    

    for max_shear, step_size in itertools.product(max_shears, step_sizes):


        period = step_size
        path = pathlib.Path(job.fn(f"aqs_experiments/max-shear-{max_shear:.2f}/temp-{temp:.4f}"))
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

        quarter_period = int(np.round(max_shear / step_size))

        period_steps = int(4 * quarter_period)

        xy_seg = np.linspace(0, max_shear, int(max_shear / step_size) + 1)

        xy = np.concatenate([xy_seg[1:], xy_seg[:-1][::-1], -xy_seg[1:], -xy_seg[:-1][::-1]])

        assert len(xy) == period_steps

        steps_between_dumps = int(period_steps / dumps)

        do_dumps = np.zeros(period_steps, dtype=bool)
        do_dumps[steps_between_dumps-1::steps_between_dumps] = True

        # make sure period_steps is a multiple of dumps
        assert period_steps % dumps == 0

        try:
            sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=doc["seed"])
            sim.create_state_from_gsd(job.fn("init.gsd"))
            fire, nve = _setup_fire_sim(job, sim, fire_kwargs={"dt": 1e-1, "force_tol": 1e-4, "angmom_tol": 1.0, "energy_tol": 1e-10})
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

            for iter in range(min_periods):
                pbar.set_description(f"Iteration {iter+1}/{min_periods}")
                for xy_val, do_dump in zip(xy, do_dumps):
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
                    if do_dump:
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
            

    doc["aqs_experiment_completed"] = True

@Project.operation
@Project.pre.true("initialized")
@Project.post.true("aqs_experiment_longer_completed")
def aqs_experiment_longer(job: signac.Project.Job):
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc

    dt = proj_doc["dt"]
    step_unit = proj_doc["step_unit"]
    equil_time = proj_doc["equil_time"]
    equil_steps = int(equil_time * step_unit)
    min_periods = 100 # proj_doc["min_periods"]
    # min_periods = 80
    dumps = proj_doc["dumps"]
    max_shears = proj_doc["max_shears"]

    temp = 0.0

    step_sizes = [0.001] # , 0.0001]

    print(sp["prep"])
    

    for max_shear, step_size in itertools.product(max_shears, step_sizes):


        period = step_size
        path = pathlib.Path(job.fn(f"aqs_experiments_longer/max-shear-{max_shear:.2f}/temp-{temp:.4f}"))
        os.makedirs(path.as_posix(), exist_ok=True)

        outfile = path / f"traj_period-{period}_more-fire.gsd"
        preequiled_outfile = path / f"preequiled-traj_period-{period}_more-fire.gsd"  # we might preequil in the future, but not now
        badfile = path / f"traj_period-{period}_more-fire.gsd.bad"
        print("Current file:", outfile.as_posix())


        if outfile.exists():
            file = gsd.hoomd.open(str(outfile), "rb")
            if len(file) >= min_periods * dumps:

                print("File already exists, skipping")
                continue

        quarter_period = int(np.round(max_shear / step_size))

        period_steps = int(4 * quarter_period)
        # total_steps = int(min_periods * period_steps)
        # output_period = int(period_steps / dumps)

        xy_seg = np.linspace(0, max_shear, int(max_shear / step_size) + 1)

        xy = np.concatenate([xy_seg[1:], xy_seg[:-1][::-1], -xy_seg[1:], -xy_seg[:-1][::-1]])

        assert len(xy) == period_steps

        steps_between_dumps = int(period_steps / dumps)

        do_dumps = np.zeros(period_steps, dtype=bool)
        do_dumps[steps_between_dumps-1::steps_between_dumps] = True

        # make sure period_steps is a multiple of dumps
        assert period_steps % dumps == 0

        try:
            sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=doc["seed"])
            # traj = gsd.hoomd.open(job.fn("init.gsd"))
            # print("len", len(traj))
            sim.create_state_from_gsd(job.fn("init.gsd"))

            nvt = _setup_nvt_sim(job, sim, temp=0.01)

            sim.run(0)
            sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=0.01)

            print("run equil")
            sim.run(equil_steps)

            sim.operations.integrator = None

            fire, nve = _setup_fire_sim(job, sim, fire_kwargs={"dt": 1e-1, "force_tol": 1e-4, "angmom_tol": 1.0, "energy_tol": 1e-10})
                                        # method=hoomd.md.methods.DisplacementCapped(hoomd.filter.All(), 0.1))
            # raise NotImplementedError("Fire is not yet implemented")
            print("initial_quench")
            pbar = tqdm()
            fire.reset()
            while not fire.converged:
                sim.run(10_000)
                pbar.update(1)
                forces = fire.forces[0].forces
                # print(forces)
                # print(fire.forces[0].energies)
                max_force = np.max(np.linalg.norm(forces, axis=1))
                pbar.set_postfix({"energy": fire.energy, "force": max_force})

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

            for iter in range(min_periods):
                pbar.set_description(f"Iteration {iter+1}/{min_periods}")
                for xy_val, do_dump in zip(xy, do_dumps):
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
                    if do_dump:
                        async_write_trig.activate()
                    sim.run(2)
                    pbar.update(1)
                # reset progress bar
                pbar.reset()
                # pbar.refresh()
                    
        except KeyboardInterrupt:
            raise
        except:
            raise
            print("Error during run, saving bad file")
            if outfile.exists():
                shutil.move(outfile.as_posix(), badfile.as_posix())
        finally:
            try:
                logger.remove(thermo)
                sim.operations.computes.clear()
                del thermo
                print("Finished run")
            except NameError:
                print("Something weird happened, but run is over")
            

    doc["aqs_experiment_longer_completed"] = True

@Project.operation
@Project.pre.true("initialized")
@Project.post.true("aqs_preshear_experiment_completed")
def aqs_experiment_preshear(job: signac.Project.Job):
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
    max_shears = proj_doc["max_shears"]

    temp = 0.0

    step_sizes = [0.001] # , 0.0001]

    coarse_step_size = 0.01
    preshear_max_shear = 0.5
    

    for max_shear, step_size in itertools.product(max_shears, step_sizes):

        preshear_cycles = 1
        period = step_size
        path = pathlib.Path(job.fn(f"aqs_experiments_preshear/max-shear-{max_shear:.2f}/temp-{temp:.4f}"))
        os.makedirs(path.as_posix(), exist_ok=True)

        outfile = path / f"traj_period-{period}_preshear-{preshear_cycles}.gsd"
        preequiled_outfile = path / f"preequiled-traj_period-{period}_preshear-{preshear_cycles}.gsd"  # we might preequil in the future, but not now
        badfile = path / f"traj_period-{period}_preshear-{preshear_cycles}.gsd.bad"
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

        quarter_period = int(np.round(max_shear / step_size))

        period_steps = int(4 * quarter_period)
        # total_steps = int(min_periods * period_steps)
        # output_period = int(period_steps / dumps)

        xy_seg = np.linspace(0, preshear_max_shear, int(preshear_max_shear / coarse_step_size) + 1)

        preshear_xy = np.concatenate([xy_seg[1:], xy_seg[:-1][::-1], -xy_seg[1:], -xy_seg[:-1][::-1]])

        steps_between_dumps = int(len(preshear_xy) / dumps)

        coarse_do_dumps = np.zeros(len(preshear_xy), dtype=bool)
        coarse_do_dumps[steps_between_dumps-1::steps_between_dumps] = True


        xy_seg = np.linspace(0, max_shear, int(max_shear / step_size) + 1)

        xy = np.concatenate([xy_seg[1:], xy_seg[:-1][::-1], -xy_seg[1:], -xy_seg[:-1][::-1]])

        assert len(xy) == period_steps

        steps_between_dumps = int(period_steps / dumps)

        do_dumps = np.zeros(period_steps, dtype=bool)
        do_dumps[steps_between_dumps-1::steps_between_dumps] = True

        # make sure period_steps is a multiple of dumps
        assert period_steps % dumps == 0

        try:
            sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=doc["seed"])
            sim.create_state_from_gsd(job.fn("init.gsd"))
            fire, nve = _setup_fire_sim(job, sim)
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

            # pbar = tqdm(total=len(xy))

            for iter in range(preshear_cycles):
                # pbar.set_description(f"Iteration {iter+1}/{min_periods}")
                for xy_val, do_dump in zip(preshear_xy, coarse_do_dumps):
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
                    if do_dump:
                        async_write_trig.activate()
                    sim.run(2)
                    # pbar.update(1)
                # reset progress bar
                # pbar.reset()
                # pbar.refresh()

            pbar = tqdm(total=len(xy))

            for iter in range(min_periods):
                pbar.set_description(f"Iteration {iter+1}/{min_periods}")
                for xy_val, do_dump in zip(xy, do_dumps):
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
                    if do_dump:
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
            

    doc["aqs_preshear_experiment_completed"] = True


@Project.operation
@Project.pre.true("initialized")
@Project.post.true("aqs_therm_experiment_completed")
def aqs_therm_shear_experiment(job: signac.Project.Job):
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc

    dt = proj_doc["dt"]
    step_unit = proj_doc["step_unit"]
    equil_time = proj_doc["equil_time"]
    equil_steps = int(equil_time * step_unit)
    min_periods = proj_doc["min_periods"] * 4
    dumps = proj_doc["dumps"]
    # period_times = proj_doc["period_times"]
    period_times = [1000.0]  # only do long timestep, others don't matter
    max_shears = proj_doc["max_shears"]

    temps = [doc["temps"][-1]*0.1, doc["temps"][-1]*0.01]  # 0.1 * Tg and 0.01 * Tg

    for temp, period, max_shear in itertools.product(temps, period_times, max_shears):

        path = pathlib.Path(job.fn(f"aqs_therm_experiments/max-shear-{max_shear:.2f}/temp-{temp:.4e}"))
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
            

    doc["aqs_therm_experiment_completed"] = True


@Project.operation
@Project.pre.true("initialized")
@Project.post.true("longer_experiment_completed")
def longer_shear_experiment(job: signac.Project.Job):
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc

    dt = proj_doc["dt"]
    step_unit = proj_doc["step_unit"]
    equil_time = proj_doc["equil_time"]
    equil_steps = int(equil_time * step_unit)
    min_periods = proj_doc["min_periods"] * 10
    dumps = proj_doc["dumps"]
    # period_times = proj_doc["period_times"]
    period_times = [1000.0]  # only do long timestep, others don't matter
    # max_shears = proj_doc["max_shears"]
    # max_shears.append(0.04)
    # max_shears.append(0.06)
    # max_shears.append(0.07)
    max_shears = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.04, 0.06, 0.07]

    # temps = doc["temps"]
    # temps.append(doc["temps"][-1]*0.1)
    # temps.append(doc["temps"][-1]*0.01)
    temps = [0.01983645726339585, 0.001983645726339585]  # 0.00019836457263395848

    # , 0.049591143158489615, 0.09918228631697923

    # 0.14877342947546884, 0.19836457263395846, # removed this because data
    # just isn't interesting out here
    
    # temps = [doc["temps"][-1]*0.1, doc["temps"][-1]*0.01]  # 0.1 * Tg and 0.01 * Tg

    for temp, period, max_shear in itertools.product(temps, period_times, max_shears):

        path = pathlib.Path(job.fn(f"longer_experiments/max-shear-{max_shear:.2f}/temp-{temp:.4e}"))
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
            

    doc["longer_experiment_completed"] = True

@Project.operation
@Project.pre.true("initialized")
@Project.post.true("longest_experiment_completed")
def longest_shear_experiment(job: signac.Project.Job):
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc

    dt = proj_doc["dt"]
    step_unit = proj_doc["step_unit"]
    equil_time = proj_doc["equil_time"]
    equil_steps = int(equil_time * step_unit)
    min_periods = proj_doc["min_periods"] * 10
    dumps = 20
    print(dumps)
    # period_times = proj_doc["period_times"]
    period_times = [1000.0]  # only do long timestep, others don't matter
    # max_shears = proj_doc["max_shears"]
    # max_shears.append(0.04)
    # max_shears.append(0.06)
    # max_shears.append(0.07)
    max_shears = [0.05]

    # temps = doc["temps"]
    # temps.append(doc["temps"][-1]*0.1)
    # temps.append(doc["temps"][-1]*0.01)
    temps = [0.001983645726339585]  # 0.00019836457263395848 # 0.01983645726339585, 

    # , 0.049591143158489615, 0.09918228631697923

    # 0.14877342947546884, 0.19836457263395846, # removed this because data
    # just isn't interesting out here
    
    # temps = [doc["temps"][-1]*0.1, doc["temps"][-1]*0.01]  # 0.1 * Tg and 0.01 * Tg

    for temp, period, max_shear in itertools.product(temps, period_times, max_shears):

        path = pathlib.Path(job.fn(f"longest_experiments/max-shear-{max_shear:.2f}/temp-{temp:.4e}"))
        # start with exisiting "longer" files
        start_path = job.fn(f"longer_experiments/max-shear-{max_shear:.2f}/temp-{temp:.4e}/traj_period-{period}.gsd")
        os.makedirs(path.as_posix(), exist_ok=True)

        period_steps = int(period * step_unit)
        total_steps = int(min_periods * period_steps)
        output_period = int(period_steps / dumps)

        # make sure period_steps is a multiple of dumps
        assert period_steps % dumps == 0

        try:
            sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=doc["seed"])
            sim.create_state_from_gsd(start_path)
            if temp == 0.0:
                fire, nve = _setup_fire_sim(job, sim)
                raise NotImplementedError("Fire is not yet implemented")
                while not fire.converged:
                    sim.run(1000)
            else:
                nvt = _setup_nvt_sim(job, sim, temp=temp)

                sim.run(0)
                # sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=temp)

            print("initial state box_xy:", sim.state.box.xy)

            thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
            logger = hoomd.logging.Logger()
            logger.add(thermo, quantities=["pressure", "pressure_tensor", "kinetic_temperature"])
            sim.operations.computes.clear()
            sim.operations.computes.append(thermo)

            sim.operations.updaters.clear()
            trigger = hoomd.trigger.Periodic(1)
            variant = methods.Oscillate(1, sim.timestep, period_steps)
            old_box = sim.state.box
            new_box = copy.deepcopy(old_box)
            old_box.xy = -max_shear
            new_box.xy = max_shear
            updater = hoomd.update.BoxResize(trigger, old_box, new_box, variant)
            sim.operations.updaters.append(updater)

            for iter in range(10):

                outfile = path / f"traj_period-{period}_iter-{iter}.gsd"
                badfile = path / f"traj_period-{period}_iter-{iter}.gsd.bad"
                print("Current file:", outfile.as_posix())

                if badfile.exists():
                    print("Bad file exists, must rerun")
                    

                if outfile.exists():
                    file = gsd.hoomd.open(str(outfile), "rb")
                    if len(file) >= min_periods * dumps:

                        print("File already exists, skipping")
                        # load last step as input for next iter
                        old_snap = gsd.hoomd.open(outfile.as_posix(), "rb")[-1]
                        # convert to hoomd snapshot from gsd
                        hoomd_snap = hoomd.Snapshot.from_gsd_snapshot(old_snap, sim.device.communicator)
                        sim.state.set_snapshot(hoomd_snap)
                        continue
                    else:
                        print("File exists, but is incomplete, continuing")


                gsd_writer = hoomd.write.GSD(hoomd.trigger.Periodic(output_period), outfile, mode="wb", log=logger)
                sim.operations.writers.clear()
                sim.operations.writers.append(gsd_writer)

                sim.run(total_steps, write_at_start=False)
            
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
            

    doc["longest_experiment_completed"] = True

@Project.operation
@Project.pre.true("initialized")
@Project.post.true("no_shear_experiment_completed")
def no_shear_experiment(job: signac.Project.Job):
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc

    dt = proj_doc["dt"]
    step_unit = proj_doc["step_unit"]
    equil_time = proj_doc["equil_time"]
    equil_steps = int(equil_time * step_unit)
    min_periods = 10
    dumps = proj_doc["dumps"]
    # period_times = proj_doc["period_times"]
    period_times = [1000.0]  # only do long timestep, others don't matter
    # max_shears = proj_doc["max_shears"]
    # max_shears.append(0.04)
    # max_shears.append(0.06)
    # max_shears.append(0.07)
    max_shears = [0.0]

    Tg = 0.1983645726339585
    temps = np.array([0.01, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.65, 0.75, 0.9, 1.0, 0.0, 0.0]) * Tg
    temps[-2] = 0.3
    temps[-1] = 0.4

    # , 0.049591143158489615, 0.09918228631697923

    # 0.14877342947546884, 0.19836457263395846, # removed this because data
    # just isn't interesting out here
    
    # temps = [doc["temps"][-1]*0.1, doc["temps"][-1]*0.01]  # 0.1 * Tg and 0.01 * Tg

    for temp, period, max_shear in itertools.product(temps, period_times, max_shears):

        path = pathlib.Path(job.fn(f"no_shear_experiments/max-shear-{max_shear:.2f}/temp-{temp:.4e}"))
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

            # sim.operations.updaters.clear()
            # trigger = hoomd.trigger.Periodic(1)
            # variant = methods.Oscillate(1, sim.timestep, period_steps)
            # old_box = sim.state.box
            # new_box = copy.deepcopy(old_box)
            # old_box.xy = -max_shear
            # new_box.xy = max_shear
            # updater = hoomd.update.BoxResize(trigger, old_box, new_box, variant)
            # sim.operations.updaters.append(updater)

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
            

    doc["no_shear_experiment_completed"] = True


@Project.operation
@Project.pre.true("initialized")
@Project.post.true("no_shear_experiment_short_completed")
def no_shear_short_experiment(job: signac.Project.Job):
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc

    dt = proj_doc["dt"]
    step_unit = proj_doc["step_unit"]
    equil_time = proj_doc["equil_time"]
    equil_steps = int(equil_time * step_unit)
    min_periods = 4
    record_steps = int(200 * step_unit)
    dumps = proj_doc["dumps"]
    period_times = [5000.0]  # only do long timestep, others don't matter

    max_shears = [0.0]

    Tg = 0.1983645726339585
    temps = np.array([0.25, 0.5, 0.75, 1.0, 0.0, 0.0]) * Tg
    temps[-2] = 0.3
    temps[-1] = 0.4

    for temp, period, max_shear in itertools.product(temps, period_times, max_shears):

        path = pathlib.Path(job.fn(f"no_shear_short_experiments/max-shear-{max_shear:.2f}/temp-{temp:.4e}"))
        os.makedirs(path.as_posix(), exist_ok=True)

        period_steps = int(period * step_unit)


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

            missing_files = False
            for i in range(min_periods):
                outfile = path / f"traj_period-{period}_iter-{i}.gsd"
                if not outfile.exists():
                    missing_files = True
                    break

            if not missing_files:
                print("All files exist, skipping")
                continue
            for i in range(min_periods):
                outfile = path / f"traj_period-{period}_iter-{i}.gsd"
                badfile = path / f"traj_period-{period}_iter-{i}.gsd.bad"
                gsd_writer = hoomd.write.GSD(hoomd.trigger.Periodic(step_unit), outfile, mode="wb", log=logger)
                sim.operations.writers.clear()
                sim.operations.writers.append(gsd_writer)

                sim.run(record_steps)
                sim.operations.writers.clear()
                sim.run(period_steps - record_steps)
            
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
            

    doc["no_shear_experiment_short_completed"] = True


@Project.operation
@Project.pre.true("initialized")
@Project.post.true("heavy_experiment_completed")
def heavy_shear_experiment(job: signac.Project.Job):
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
    # period_times = proj_doc["period_times"]
    period_times = [1000.0]  # only do long timestep, others don't matter
    # max_shears = proj_doc["max_shears"]
    # max_shears.append(0.04)
    # max_shears.append(0.06)
    # max_shears.append(0.07)
    max_shears = [0.15, 0.20, 0.3, 0.4, 0.5]

    # temps = doc["temps"]
    # temps.append(doc["temps"][-1]*0.1)
    # temps.append(doc["temps"][-1]*0.01)
    temps = [0.01983645726339585, 0.001983645726339585, 0.049591143158489615, 0.09918228631697923]  # 0.00019836457263395848

    # 0.14877342947546884, 0.19836457263395846, # removed this because data
    # just isn't interesting out here

    # temps = [doc["temps"][-1]*0.1, doc["temps"][-1]*0.01]  # 0.1 * Tg and 0.01 * Tg

    for temp, period, max_shear in itertools.product(temps, period_times, max_shears):

        path = pathlib.Path(job.fn(f"heavy_experiments/max-shear-{max_shear:.2f}/temp-{temp:.4e}"))
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
            

    doc["heavy_experiment_completed"] = True


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
    for run in runs:
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
@Project.pre.after(no_shear_experiment)
@Project.post.true("no_shear_fire_applied")
def no_shear_fire_minimize(job: signac.Project.Job):
    sp = job.sp
    doc = job.doc
    print(job)

    sim = hoomd.Simulation(hoomd.device.GPU(), seed=doc["seed"])
    sim.create_state_from_gsd(job.fn("init.gsd"))
    fire, nve = _setup_fire_sim(job, sim)

    # setup details for sims
    runs = glob.glob(job.fn(f"no_shear_experiments/max-shear-*/temp-*/traj_period-*.gsd"))
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
    job.doc["no_shear_fire_applied"] = True


@Project.operation
@Project.pre.after(no_shear_experiment)
@Project.post.true("no_shear_short_fire_applied")
def no_shear_short_fire_minimize(job: signac.Project.Job):
    sp = job.sp
    doc = job.doc
    print(job)

    sim = hoomd.Simulation(hoomd.device.GPU(), seed=doc["seed"])
    sim.create_state_from_gsd(job.fn("init.gsd"))
    fire, nve = _setup_fire_sim(job, sim)

    # setup details for sims
    runs = glob.glob(job.fn(f"no_shear_short_experiments/max-shear-*/temp-*/traj_period-*.gsd"))
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
    job.doc["no_shear_short_fire_applied"] = True


@Project.operation
@Project.pre.after(longer_shear_experiment)
@Project.post.true("longer_fire_applied")
def longer_fire_minimize(job: signac.Project.Job):
    sp = job.sp
    doc = job.doc
    print(job)

    sim = hoomd.Simulation(hoomd.device.GPU(), seed=doc["seed"])
    sim.create_state_from_gsd(job.fn("init.gsd"))
    fire, nve = _setup_fire_sim(job, sim)

    # setup details for sims
    runs = glob.glob(job.fn(f"longer_experiments/max-shear-*/temp-*/traj_period-*.gsd"))
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
    job.doc["longer_fire_applied"] = True

@Project.operation
@Project.pre.after(longer_shear_experiment)
@Project.post.true("longest_fire_applied")
def longest_fire_minimize(job: signac.Project.Job):
    sp = job.sp
    doc = job.doc
    print(job)

    sim = hoomd.Simulation(hoomd.device.GPU(), seed=doc["seed"])
    sim.create_state_from_gsd(job.fn("init.gsd"))
    fire, nve = _setup_fire_sim(job, sim)

    # setup details for sims
    runs = glob.glob(job.fn(f"longest_experiments/max-shear-*/temp-*/traj_period-*.gsd"))
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
    job.doc["longest_fire_applied"] = True

@Project.operation
@Project.pre.after(aqs_therm_shear_experiment)
@Project.post.true("aqs_therm_fire_applied")
def aqs_therm_fire_minimize(job: signac.Project.Job):
    sp = job.sp
    doc = job.doc
    print(job)

    sim = hoomd.Simulation(hoomd.device.GPU(), seed=doc["seed"])
    sim.create_state_from_gsd(job.fn("init.gsd"))
    fire, nve = _setup_fire_sim(job, sim)

    # setup details for sims
    runs = glob.glob(job.fn(f"aqs_therm_experiments/max-shear-*/temp-*/traj_period-*.gsd"))
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
    job.doc["aqs_therm_fire_applied"] = True


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

@Project.operation
@Project.pre.after(aqs_therm_shear_experiment)
@Project.post.true("trad_excess_entropy_computed")
def trad_excess_entropy(job: signac.Project.Job):
    print(job)
    
    experiments = sorted(glob.glob(job.fn("aqs_therm_experiments/*/*/traj_period-*.gsd")))
    if len(experiments) == 0:
        return
    for exper in experiments:
        max_shear = utils.extract_between(exper, "max-shear-", "/")
        period = utils.extract_between(exper, "period-", ".gsd")
        temp = utils.extract_between(exper, "temp-", "/")
        df_path = f"aqs_therm_experiments/max-shear-{max_shear}/temp-{temp}/excess-entropy-small_period-{period}.parquet"
        
        if float(period) != 1000.0:
            continue

        # if job.isfile(df_path):
        #     continue

        traj = gsd.hoomd.open(exper)

        
        print(max_shear, period, temp)

        entropy = []
        # rdfs = []
        rdf_aas = []
        rdf_abs = []
        rdf_bbs = []
        rs = []
        frames = []
        # types = []
        start = 60
        end = 80
        cycle_start_idx = lambda i: -1 + i*40
        for frame in tqdm(range(cycle_start_idx(start), cycle_start_idx(end))):

            frames.append(frame)

            snap = traj[frame]

            typeids = snap.particles.typeid
            # types.append(typeids)

            box = snap.configuration.box
            pos_a = snap.particles.position[typeids == 0]
            pos_b = snap.particles.position[typeids == 1]

            query_a = freud.locality.AABBQuery.from_system((box, pos_a))
            query_b = freud.locality.AABBQuery.from_system((box, pos_b))

            query_args = {"r_min": 0.01, "r_max": 6.0}
            nlist_aa = query_a.query(pos_a, query_args).toNeighborList()
            nlist_ab = query_b.query(pos_a, query_args).toNeighborList()
            nlist_bb = query_b.query(pos_b, query_args).toNeighborList()
            r, rdf_aa = compute_rdf(nlist_aa, bins=120, r_max=6)
            _, rdf_ab = compute_rdf(nlist_ab, bins=120, r_max=6)
            _, rdf_bb = compute_rdf(nlist_bb, bins=120, r_max=6)

            rdf_aas.append(rdf_aa)
            rdf_abs.append(rdf_ab)
            rdf_bbs.append(rdf_bb)
            rs.append(r)
            
            entropy.append(binary_excess_entropy(r, rdf_aa, rdf_ab, rdf_bb, 0.6, 0.4, r[1] - r[0]))
            # break
        # break
        dataset = pl.DataFrame({"frame": frames, "r": rs, "rdf_aa": rdf_aas, "rdf_ab": rdf_abs, "rdf_bb": rdf_bbs, "entropy": entropy})
        # # dataset = dataset.explode(["id", "soft"]).reset_index(drop=True)
        dataset.write_parquet(job.fn(df_path), use_pyarrow=True)
    job.doc["trad_excess_entropy_computed"] = True

if __name__ == "__main__":
    Project.get_project(root=config["root"]).main()