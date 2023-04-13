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
    elif isinstance(pot, signac.synced_collections.backends.collection_json.JSONAttrList):
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
    
    default_fire_kwargs = {"dt": 1e-2, "force_tol": 1e-3, "angmom_tol": 1.0, "energy_tol": 1e-3}
    
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

    nvt = hoomd.md.methods.NVE(hoomd.filter.All())
    fire.forces = [force]
    fire.methods = [nvt]

    sim.operations.integrator = fire

    return fire, nvt


def _write_quench_dynamics_to_job(job, action, temp):
    job.data[f"quench/temp_{temp:.3f}/tsteps".replace(".", "_")] = np.array(action.tsteps)
    job.data[f"quench/temp_{temp:.3f}/msds".replace(".", "_")] = np.array(action.msds)
    job.data[f"quench/temp_{temp:.3f}/sisfs".replace(".", "_")] = np.array(action.sisfs)
    job.data[f"quench/temp_{temp:.3f}/alphas".replace(".", "_")] = np.array(action.alphas)
    job.data[f"quench/temp_{temp:.3f}/Ds".replace(".", "_")] = np.array(action.Ds)


def _write_fine_dynamics_to_job(job, action, temp):
    job.data[f"fine/temp_{temp:.3f}/tsteps".replace(".", "_")] = np.array(action.tsteps)
    job.data[f"fine/temp_{temp:.3f}/msds".replace(".", "_")] = np.array(action.msds)
    job.data[f"fine/temp_{temp:.3f}/sisfs".replace(".", "_")] = np.array(action.sisfs)
    job.data[f"fine/temp_{temp:.3f}/alphas".replace(".", "_")] = np.array(action.alphas)
    job.data[f"fine/temp_{temp:.3f}/Ds".replace(".", "_")] = np.array(action.Ds)


def roundup(x):
    return 10**math.ceil(math.log10(x))


def rounddown(x):
    return 10**math.floor(math.log10(x))


@Project.operation
@Project.post.true("primary_equilibration_done")
def primary_equil_quench(job: signac.Project.Job):
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc

    step_unit = proj_doc["step_unit"]
    equil_time = proj_doc["equil_time"]
    max_alpha_time = proj_doc["max_alpha_time"]
    run_steps = proj_doc["run_time"] * step_unit
    alpha_iters = proj_doc["alpha_iters"]

    marker = pathlib.Path(job.fn("quench/done"))

    if marker.exists():
        print("Primary equilibration already done")
        return

    # initialize first simualtion
    if not job.isfile("init.gsd"):
        print("Initializing simulation")
        sim = prep.quick_sim(sp["N"], sp["rho"], hoomd.device.GPU(), ratios=[
                             80, 20], diams=[1.0, 0.88], seed=doc["seed"])
        nvt = _setup_nvt_sim(job, sim)

        sim.run(0)
        nvt.thermalize_thermostat_dof()
        print("Running initial equilibration")
        sim.run(100 * step_unit)

        hoomd.write.GSD.write(sim.state, job.fn("init.gsd"))

        del sim, nvt
        gc.collect()
    else:
        print("init.gsd found, skipping initialization")

    sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=doc["seed"])
    sim.create_state_from_gsd(filename=job.fn("init.gsd"))
    nvt = _setup_nvt_sim(job, sim)

    os.makedirs(job.fn("quench"), exist_ok=True)

    try:
        for temp in doc["temp_steps"][::-1]:
            final_traj_name = f"quench/equil_temp-{temp:.3f}.gsd"
            final_traj = job.fn(final_traj_name)
            if job.isfile(final_traj_name):
                print(f"Trajectory found for temp {temp}, skipping")
                del sim, nvt
                gc.collect()
                sim = hoomd.Simulation(
                    device=hoomd.device.GPU(), seed=doc["seed"])
                sim.create_state_from_gsd(filename=final_traj)
                nvt = _setup_nvt_sim(job, sim, temp=temp)
                continue
            else:
                print("Running temperature step ", temp)
                nvt.kT = temp
            print("Temperature set to ", nvt.kT(sim.timestep))

            action = methods.VerifyEquilibrium(max_alpha_time=max_alpha_time)
            custom_action = hoomd.update.CustomUpdater(
                hoomd.trigger.Periodic(run_steps), action)
            sim.operations.updaters.clear()
            sim.operations.updaters.append(custom_action)

            tmp_traj = job.fn(f"quench/_equil_temp-{temp:.3f}.gsd")

            sim.run(equil_time * step_unit)

            # hoomd.write.GSD.write(sim.state, final_traj)
            gsd_writer = hoomd.write.GSD(filename=tmp_traj,
                                    trigger=hoomd.trigger.Periodic(run_steps, phase=sim.timestep),
                                    mode='wb',
                                    )
            sim.operations.writers.clear()
            sim.operations.writers.append(gsd_writer)

            while len(action.alphas) < alpha_iters:
                sim.run(run_steps)
            print("Alpha relaxation times:", action.alphas[-5:])
            print("Diffusion coefficients:", action.Ds[-5:])
            _write_quench_dynamics_to_job(job, action, temp)
            shutil.move(tmp_traj, final_traj)

    except RuntimeError as e:
        assert str(e) == "Alpha relaxation time is too long."
        marker.touch()
        not_equil = job.fn(f"quench/not-equil_temp-{temp:.3f}.gsd")
        shutil.move(tmp_traj, not_equil)
    except Exception as e:
        raise RuntimeError("Simulation failed with exception", str(e))
    finally:
        print("Primary equilibration done")
        doc["primary_equilibration_done"] = True


@Project.operation
@Project.pre.after(primary_equil_quench)
@Project.post.true("fine_quench_done")
def sc_fine_quench(job: signac.Project.Job):
    """Sample temperatures in the supercooled regime more finely"""
    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc

    step_unit = proj_doc["step_unit"]
    equil_time = proj_doc["equil_time"]
    max_alpha_time = 4_000
    run_steps = step_unit  # this will be adjusted as the simulations progress
    alpha_iters = 10

    marker = pathlib.Path(job.fn("fine/done"))

    if marker.exists():
        print("fine quench already done")
        return

    sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=doc["seed"])
    sim.create_state_from_gsd(filename=job.fn("init.gsd"))
    nvt = _setup_nvt_sim(job, sim)

    os.makedirs(job.fn("fine"), exist_ok=True)

    # only look at the last four temperatures
    old_sims = sorted(glob.glob(job.fn("quench/equil_temp-*.gsd")), key=lambda x: float(utils.extract_between(x, "temp-", ".gsd")))
    # print(old_sims)
    start_job = old_sims[4]
    start_temp = float(utils.extract_between(start_job, "temp-", ".gsd"))
    _next_temp = float(utils.extract_between(old_sims[3], "temp-", ".gsd"))
    delta_temp = (start_temp - _next_temp) / 2

    # print(start_temp, _next_temp, delta_temp)

    # assert False

    assert delta_temp > 0.0
    
    temp = start_temp

    sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=doc["seed"])
    sim.create_state_from_gsd(filename=start_job)
    nvt = _setup_nvt_sim(job, sim)

    try:
        while temp > 0.0:
            final_traj_name = f"fine/equil_temp-{temp:.3f}.gsd"
            final_traj = job.fn(final_traj_name)
            print("Run steps:", run_steps)
            if job.isfile(final_traj_name):
                print(f"Trajectory found for temp {temp}, skipping")
                del sim, nvt
                gc.collect()
                sim = hoomd.Simulation(
                    device=hoomd.device.GPU(), seed=doc["seed"])
                sim.create_state_from_gsd(filename=final_traj)
                nvt = _setup_nvt_sim(job, sim, temp=temp)
                with job.data:
                    run_steps = rounddown(np.mean(job.data[f"fine/temp_{temp:.3f}/alphas".replace(".", "_")][-5:])) * step_unit
                temp -= delta_temp
                continue
            else:
                print("Running temperature step ", temp)
                nvt.kT = temp
            print("Temperature set to ", nvt.kT(sim.timestep))

            action = methods.VerifyEquilibrium(max_alpha_time=max_alpha_time)
            custom_action = hoomd.update.CustomUpdater(
                hoomd.trigger.Periodic(run_steps), action)
            sim.operations.updaters.clear()
            sim.operations.updaters.append(custom_action)

            tmp_traj = job.fn(f"fine/_equil_temp-{temp:.3f}.gsd")

            sim.run(equil_time * step_unit)

            # hoomd.write.GSD.write(sim.state, final_traj)
            gsd_writer = hoomd.write.GSD(filename=tmp_traj,
                                    trigger=hoomd.trigger.Periodic(run_steps, phase=sim.timestep),
                                    mode='wb',
                                    )
            sim.operations.writers.clear()
            sim.operations.writers.append(gsd_writer)

            while len(action.alphas) < alpha_iters:
                sim.run(run_steps)
            print("Alpha relaxation times:", action.alphas[-5:])
            print("Diffusion coefficients:", action.Ds[-5:])
            _write_fine_dynamics_to_job(job, action, temp)
            shutil.move(tmp_traj, final_traj)

            with job.data:
                run_steps = rounddown(np.mean(job.data[f"fine/temp_{temp:.3f}/alphas".replace(".", "_")][-5:])) * step_unit
            temp -= delta_temp

    except RuntimeError as e:
        assert str(e) == "Alpha relaxation time is too long."
        marker.touch()
        not_equil = job.fn(f"fine/not-equil_temp-{temp:.3f}.gsd")
        shutil.move(tmp_traj, not_equil)
    except Exception as e:
        raise RuntimeError("Simulation failed with exception", str(e))

    print("Fine quench done")
    doc["fine_quench_done"] = True


@Project.operation
@Project.pre.after(primary_equil_quench)
@Project.post.true("secondary_equilibration_done")
def secondary_equil(job: signac.Project.Job):

    doc = job.doc
    sp = job.sp
    project = job._project
    proj_doc = project.doc

    step_unit = proj_doc["step_unit"]
    run_steps = 1000 * step_unit

    os.makedirs(job.fn("equil"), exist_ok=True)

    # only analysis the last four runs
    quench_runs = sorted(glob.glob(job.fn("quench/equil_temp-*.gsd")))[:4]
    for run in quench_runs:
        temp = float(utils.extract_between(run, "temp-", ".gsd"))
        new_traj = pathlib.Path(run.replace("quench", "equil"))
        if new_traj.exists():
            print(f"Trajectory found for temp {temp}, skipping")
            continue
        else:
            print("Running temperature step ", temp)

        # run simulation
        sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=doc["seed"])
        sim.create_state_from_gsd(filename=run)
        nvt = _setup_nvt_sim(job, sim, temp=temp)
        print("Temperature set to ", nvt.kT(sim.timestep))

        tmp_traj = str(new_traj).replace("equil_temp", "_equil_temp")
        assert tmp_traj != new_traj.as_posix()

        gsd_writer = hoomd.write.GSD(filename=tmp_traj,
                                    trigger=hoomd.trigger.Periodic(step_unit, phase=sim.timestep),
                                    mode='wb',
                                    )
        sim.operations.writers.clear()
        sim.operations.writers.append(gsd_writer)

        sim.run(run_steps)

        shutil.move(tmp_traj, new_traj.as_posix())

    print("Secondary equilibration done")
    doc["secondary_equilibration_done"] = True


@Project.operation
@Project.pre.after(secondary_equil)
@Project.post.true("fire_applied")
def fire_minimize(job: signac.Project.Job):
    sp = job.sp
    doc = job.doc

    os.makedirs(job.fn("analysis"), exist_ok=True)

    sim = hoomd.Simulation(hoomd.device.GPU(), seed=doc["seed"])
    sim.create_state_from_gsd(job.fn("init.gsd"))
    fire, nve = _setup_fire_sim(job, sim)

    # setup details for sims
    runs = glob.glob(job.fn("*equil/equil_temp-*.gsd"))
    for run in runs:
        print(run)
        traj = gsd.hoomd.open(run)
        fire_traj = run.replace("equil/equil_temp", "analysis/long_fire_temp")

        assert fire_traj != traj

        if os.path.isfile(fire_traj):
            tmp_traj = gsd.hoomd.open(fire_traj)
            len_traj = len(tmp_traj)
            del tmp_traj
            if len_traj == 100:
                continue

        methods.fire_minimize_frames(
            sim,
            traj,
            fire_traj,
            fire_steps=1000
        )

    job.doc["fire_applied"] = True

@Project.operation
@Project.pre.after(fire_minimize)
@Project.post.true('training_sfs_computed')
def compute_training_sfs(job: signac.Project.Job):
    """Compute stucture functions for training."""

    print("Job ID:", job.id)
    # only one run should be used
    runs = sorted(glob.glob(job.fn("analysis/fire_temp-*.gsd")))
    run = runs[0]
    traj = gsd.hoomd.open(run)
    ref_snap = traj[0]
    ref_pos = ref_snap.particles.position.copy()
    box = freud.box.Box.from_box(ref_snap.configuration.box)
    pos_shape = ref_pos.shape
    pos = np.zeros((len(traj), *pos_shape), dtype=np.float32)
    pos[0] = ref_pos
    for i, snap in enumerate(traj[1:]):
        next_pos = snap.particles.position.copy()
        pos[i+1] = box.wrap(next_pos - ref_pos) + pos[i]
        ref_pos = next_pos

    # compute phop
    print("Computing phop")
    phop = _schmeud.dynamics.p_hop(pos, 11)

    soft_hard_indices = ml.group_hard_soft_by_cutoffs(phop, 0.05, 0.2, distance=100, hard_distance=800, sub_slice=slice(0, len(phop)))

    df_frame = []
    df_tag = []
    df_type = []
    df_soft = []
    df_phop = []
    df_sf = []


    print("Computing structure functions")
    for frame, soft_indices, hard_indices in soft_hard_indices:
        soft_indices = np.array(soft_indices, dtype=np.uint32)
        hard_indices = np.array(hard_indices, dtype=np.uint32)
        snap: gsd.hoomd.Snapshot = traj[int(frame)]
        query_indices = np.concatenate([soft_indices, hard_indices])
        perm = query_indices.argsort()
        # sort arrays by query indices
        query_indices = query_indices[perm]
        truths = np.concatenate([np.ones_like(soft_indices, dtype=bool), np.zeros_like(hard_indices, dtype=bool)])[perm]
        # get sfs
        sfs = ml.compute_structure_functions_snap(snap, query_indices)

        df_frame.append(frame*np.ones_like(query_indices, dtype=np.uint32))
        df_tag.append(query_indices)
        df_type.append(snap.particles.typeid[query_indices])
        df_soft.append(truths)
        df_phop.append(phop[frame][query_indices])
        df_sf.append(sfs)

    df = pd.DataFrame(
        {
            "frame": np.concatenate(df_frame),
            "tag": np.concatenate(df_tag),
            "type": np.concatenate(df_type),
            "soft": np.concatenate(df_soft),
            "phop": np.concatenate(df_phop),
            "sf": list(np.concatenate(df_sf)),
        }
    )

    out_file = run.replace("fire", "sfs").replace(".gsd", ".parquet")
    assert out_file != run

    df.to_parquet(out_file)

    job.doc["training_sfs_computed"] = True


@Project.operation
@Project.pre.after(compute_training_sfs)
@Project.post.true('softness_trained')
def train_softness(job: signac.Project.Job):
    print("Training softness")
    print(job.sp)
    proj: signac.Project = job._project

    os.makedirs(job.fn("pipelines"), exist_ok=True)

    df = pd.read_parquet(glob.glob(job.fn("analysis/sfs_temp-*.parquet"))[0])
    df = df[df.type == 0]

    print(df["phop"].describe())

    if df["phop"].min() > 0.2:
        return

    data_size = df['soft'].value_counts()

    # sample = 10_000
    sample = None
    if data_size.min() > 10_000:
        sample = 10_000
    for sample_seed in range(0, 3):

        if sample is not None:
            df = df.groupby('soft').apply(lambda x: x.sample(sample, random_state=sample_seed)).reset_index(drop=True)
            samples_used = sample
        else:
            df = df.groupby('soft').apply(lambda x: x.sample(data_size.min(), random_state=sample_seed)).reset_index(drop=True)
            samples_used = data_size.min()
        
        print(samples_used)

        # return
        for seed in range(10, 13):
            print(sample_seed, seed)
            pipe, acc = ml.train_hyperplane_pipeline(
                df["sf"],
                df["soft"],
                seed=seed,
                max_iter=100_000
            )

            out_file = job.fn(f"pipelines/pipe_ss-{sample_seed}_ts-{seed}.pickle")

            with open(out_file, "wb") as f:
                pickle.dump({"pipe": pipe, "acc": acc, "data_size": data_size, "samples": samples_used}, f)

    job.doc['softness_trained'] = True

@Project.operation
@Project.pre.after(train_softness)
@Project.post.true('analysis_sfs_computed')
def compute_analysis_sfs(job: signac.Project.Job):
    """Compute stucture functions for analysis."""

    print("Job ID:", job.id)

    file = glob.glob(job.fn(f"pipelines/*.pickle"))[0]
    with open(file, "rb") as f:
        pipeline = pickle.load(f)
    pipe = pipeline["pipe"]

    runs = glob.glob(job.fn("analysis/fire_temp-*.gsd"))
    for run in runs:
        print(run)
        traj = gsd.hoomd.open(run)
        ref_snap = traj[0]
        ref_pos = ref_snap.particles.position.copy()
        box = freud.box.Box.from_box(ref_snap.configuration.box)
        pos_shape = ref_pos.shape
        pos = np.zeros((len(traj), *pos_shape), dtype=np.float32)
        pos[0] = ref_pos
        for i, snap in enumerate(traj[1:]):
            next_pos = snap.particles.position.copy()
            pos[i+1] = box.wrap(next_pos - ref_pos) + pos[i]
            ref_pos = next_pos

        # compute phop
        print("Computing phop")
        phop = _schmeud.dynamics.p_hop(pos, 11)

        df_frame = []
        df_tag = []
        df_type = []
        df_phop = []
        df_sf = []
        df_soft = []

        print("Computing structure functions")
        for frame in range(0, len(phop), 5):
            snap: gsd.hoomd.Snapshot = traj[int(frame)]
            num_particles = snap.particles.N

            # get sfs
            sfs = ml.compute_structure_functions_snap(snap)
            softness = np.array(pipe.decision_function(list(sfs)))
            softness[snap.particles.typeid != 0] = np.nan

            df_frame.append(frame*np.ones(num_particles, dtype=np.uint32))
            df_tag.append(np.arange(num_particles))
            df_type.append(snap.particles.typeid)
            df_phop.append(phop[frame])
            df_sf.append(sfs)
            df_soft.append(softness)

        df = pd.DataFrame(
            {
                "frame": np.concatenate(df_frame),
                "tag": np.concatenate(df_tag),
                "type": np.concatenate(df_type),
                "phop": np.concatenate(df_phop),
                "sf": list(np.concatenate(df_sf)),
                "softness": np.concatenate(df_soft),
            }
        )

        out_file = run.replace("fire", "analysis-sfs").replace(".gsd", ".parquet")

        assert out_file != run

        df.to_parquet(out_file)

    job.doc["analysis_sfs_computed"] = True


if __name__ == "__main__":
    Project.get_project(root=config["root"]).main()
