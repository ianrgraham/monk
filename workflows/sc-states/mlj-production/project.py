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

from monk import workflow, methods, pair, utils
import schmeud._schmeud as _schmeud
from schmeud import ml

config = workflow.get_config()

class Project(flow.FlowProject):
    pass

@Project.operation
@Project.pre(lambda job: not job.doc.get('_CRYSTAL'))
@Project.post.true('fire_applied')
def fire_minimize_analysis_runs(job: signac.Project.Job):
    """Apply FIRE minimization to the runs ready for analysis."""
    sp = job.sp
    doc = job.doc

    sim = hoomd.Simulation(hoomd.device.GPU(), seed=doc["seed"])
    sim.create_state_from_gsd(job.fn("init.gsd"))

    fire = hoomd.md.minimize.FIRE(1e-2, 1e-3, 1.0, 1e-3)

    nlist = hoomd.md.nlist.Tree(buffer=0.3)
    mlj = pair.KA_ModLJ(nlist, sp["delta"])
    nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
    fire.forces = [mlj]
    fire.methods = [nve]
    sim.operations.integrator = fire

    # setup details for sims
    runs = glob.glob(job.fn("*_runs/temp-*/traj.gsd"))
    for run in runs:
        print(run)
        traj = gsd.hoomd.open(run)
        fire_traj = run.replace("traj.gsd", "fire_traj.gsd")

        assert fire_traj != traj

        if os.path.isfile(fire_traj):
            tmp_traj = gsd.hoomd.open(fire_traj)
            len_traj = len(tmp_traj)
            del tmp_traj
            if len_traj == len(traj):
                continue

        methods.fire_minimize_frames(
            sim,
            traj,
            fire_traj
        )

    job.doc["fire_applied"] = True


@Project.operation
@Project.pre.true('fire_applied')
@Project.post.true('training_sfs_computed')
def compute_training_sfs(job: signac.Project.Job):
    """Compute stucture functions for training."""

    print("Job ID:", job.id)
    # only one run should be used
    runs = glob.glob(job.fn("training_runs/*/fire_traj.gsd"))
    assert len(runs) == 1
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

    out_file = run.replace("fire_traj.gsd", "sfs.parquet")

    df.to_parquet(out_file)

    job.doc["training_sfs_computed"] = True


@flow.aggregator.groupby("delta", sort_by="delta")
@Project.operation
@Project.pre.true('training_sfs_computed')
@Project.post.true('softness_trained')
def train_softness(*jobs: signac.Project.Job):
    """Train softness for each delta."""
    print("Training softness")
    print(jobs)
    job = jobs[0]
    delta = job.sp["delta"]
    proj: signac.Project = job._project
    output_dir = proj.root_directory() + f"/softness_pipelines/delta-{delta:.1f}/"
    os.makedirs(output_dir, exist_ok=True)

    dfs = []

    for job in jobs:
        if 'softness_trained' in job.sp:
            del job.sp['softness_trained']
        dfs.append(pd.read_parquet(glob.glob(job.fn("training_runs/*/sfs.parquet"))[0]))

    print(jobs)

    df = pd.concat(dfs)
    df = df[df.type == 0]
    del dfs

    data_size = df['soft'].value_counts()

    sample = 10_000
    # sample = None
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
            pipe, acc = ml.train_hyperplane_pipeline(
                df["sf"],
                df["soft"],
                seed=seed,
                max_iter=10_000
            )

            out_file = output_dir + f"pipe_ss-{sample_seed}_ts-{seed}.pickle"

            with open(out_file, "wb") as f:
                pickle.dump({"pipe": pipe, "acc": acc, "data_size": data_size, "samples": samples_used}, f)

    for job in jobs:
        job.doc['softness_trained'] = True


@Project.operation
@Project.pre.true('softness_trained')
@Project.post.true('analysis_sfs_computed')
def compute_analysis_sfs(job: signac.Project.Job):
    """Compute stucture functions for analysis."""

    print("Job ID:", job.id)
    # only one run should be used
    runs = glob.glob(job.fn("short_runs/*/fire_traj.gsd"))
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


        print("Computing structure functions")
        for frame in range(0, len(phop), 10):
            # soft_indices = np.array(soft_indices, dtype=np.uint32)
            # hard_indices = np.array(hard_indices, dtype=np.uint32)
            snap: gsd.hoomd.Snapshot = traj[int(frame)]
            num_particles = snap.particles.N
            # query_indices = np.concatenate([soft_indices, hard_indices])
            # perm = query_indices.argsort()
            # sort arrays by query indices

            # get sfs
            sfs = ml.compute_structure_functions_snap(snap)

            df_frame.append(frame*np.ones(num_particles, dtype=np.uint32))
            df_tag.append(np.arange(num_particles))
            df_type.append(snap.particles.typeid)
            df_phop.append(phop[frame])
            df_sf.append(sfs)

        df = pd.DataFrame(
            {
                "frame": np.concatenate(df_frame),
                "tag": np.concatenate(df_tag),
                "type": np.concatenate(df_type),
                "phop": np.concatenate(df_phop),
                "sf": list(np.concatenate(df_sf)),
            }
        )

        out_file = run.replace("fire_traj.gsd", "sfs.parquet")

        df.to_parquet(out_file)

    job.doc["analysis_sfs_computed"] = True


class xyLogger:

    def __init__(self, sim: hoomd.Simulation):
        self._sim = sim

    @property
    def value(self):
        return self._sim.state.box.xy


# shear simulations
@Project.operation
# @Project.pre(lambda job: not job.doc.get('_CRYSTAL'))
@Project.pre(lambda job: job.sp["delta"] == 0.0)
@Project.post.true('const_shear_ran')
def constant_shear_runs(job: signac.Project.Job):
    """Run constant shear simulations."""
    sp = job.sp
    doc = job.doc

    print(job, sp, doc)

    project = job._project

    dt = project.doc["dt"]

    shear_rates = [1e-3, 1e-4, 1e-5]
    # instead use const number of time steps
    # max_strain = 0.40
    # step_size = 1e-1

    total_steps = 500_000

    def setup_sim(gsd_file, temp):
        sim = hoomd.Simulation(hoomd.device.GPU(), seed=doc["seed"])
        sim.create_state_from_gsd(gsd_file)

        integrator = hoomd.md.Integrator(dt)

        nlist = hoomd.md.nlist.Tree(buffer=0.3)
        mlj = pair.KA_ModLJ(nlist, sp["delta"])
        nve = hoomd.md.methods.NVT(hoomd.filter.All(), temp, 0.1)
        integrator.forces = [mlj]
        integrator.methods = [nve]
        sim.operations.integrator = integrator

        return sim

    def job_task(job, max_strain, shear_rate, temp):
        output_dir = job.fn(f"shear_runs/rate-{shear_rate}/temp-{temp}")
        if os.path.exists(output_dir + "/traj.gsd"):
            return
        os.makedirs(output_dir, exist_ok=True)

        sim = setup_sim(run, float(temp))

        sim.always_compute_pressure = True

        log = hoomd.logging.Logger()

        analyzer = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
        xy_logger = xyLogger(sim)
        sim.operations.computes.append(analyzer)

        log.add(analyzer, quantities=["pressure", "pressure_tensor", "kinetic_temperature"])
        log.add(sim, quantities=["timestep"])
        log[('box', 'xy')] = (xy_logger, 'value', 'scalar')

        box_i = sim.state.box
        box_f = copy.deepcopy(box_i)
        box_f.xy = max_strain
        
        total_steps = int(np.round(max_strain/shear_rate/dt))
        shear_update_period = 1 # int(np.round(step_size/shear_rate))
        trigger = hoomd.trigger.Periodic(shear_update_period, phase=sim.timestep)
        variant = hoomd.variant.Ramp(0, 1, sim.timestep, total_steps)
        updater = hoomd.update.BoxResize(trigger, box_i, box_f, variant)
        sim.operations.updaters.append(updater)

        write_trigger = hoomd.trigger.Periodic(10_000, phase=sim.timestep)
        tmp_file = output_dir + "/_traj.gsd"
        traj = output_dir + "/traj.gsd"
        writer = hoomd.write.GSD(filename=tmp_file, trigger=write_trigger, mode="wb", log=log)
        sim.operations.writers.append(writer)

        # print(total_steps)

        sim.run(total_steps)

        del sim, box_i, box_f, log, analyzer, xy_logger, updater, writer, trigger, variant, write_trigger

        gc.collect()

        shutil.move(tmp_file, traj)

    runs = sorted(glob.glob(job.fn("runs/temp-*/traj.gsd")))
    # print(runs)
    for run in runs[:10]:
        temp = utils.extract_between(run, "temp-", "/traj.gsd")
        print("temp:", temp)
        processes = []
        for shear_rate in shear_rates:
            max_strain = total_steps*shear_rate*dt
            process = multiprocessing.Process(target=job_task, args=(job, max_strain, shear_rate, temp))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()

    job.doc["const_shear_ran"] = True


# shear simulations
@Project.operation
# @Project.pre(lambda job: not job.doc.get('_CRYSTAL'))
@Project.pre(lambda job: job.sp["delta"] == 0.0 and job.doc.get("const_shear_ran"))
@Project.post.true("const_shear_analysis_ran")
def constant_shear_runs_analysis(job: signac.Project.Job):
    """Run constant shear simulations."""
    sp = job.sp
    doc = job.doc

    print(job, sp, doc)

    project = job._project

    dt = project.doc["dt"]

    total_steps = 100_000

    def setup_sim(gsd_file, temp):
        sim = hoomd.Simulation(hoomd.device.GPU(), seed=doc["seed"])
        sim.create_state_from_gsd(gsd_file)

        integrator = hoomd.md.Integrator(dt)

        nlist = hoomd.md.nlist.Tree(buffer=0.3)
        mlj = pair.KA_ModLJ(nlist, sp["delta"])
        nve = hoomd.md.methods.NVT(hoomd.filter.All(), temp, 0.1)
        integrator.forces = [mlj]
        integrator.methods = [nve]
        sim.operations.integrator = integrator

        return sim

    def job_task(job, max_strain, shear_rate, temp):
        output_dir = job.fn(f"shear_runs/rate-{shear_rate}/temp-{temp}")
        sim = setup_sim(run, float(temp))

        sim.always_compute_pressure = True

        log = hoomd.logging.Logger()

        analyzer = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
        xy_logger = xyLogger(sim)
        sim.operations.computes.append(analyzer)

        log.add(analyzer, quantities=["pressure", "pressure_tensor", "kinetic_temperature"])
        log.add(sim, quantities=["timestep"])
        log[('box', 'xy')] = (xy_logger, 'value', 'scalar')

        box_i = sim.state.box
        box_f = copy.deepcopy(box_i)
        box_f.xy = max_strain + box_i.xy
        
        total_steps = int(np.round(max_strain/shear_rate/dt))
        shear_update_period = 1 # int(np.round(step_size/shear_rate))
        trigger = hoomd.trigger.Periodic(shear_update_period, phase=sim.timestep)
        variant = hoomd.variant.Ramp(0, 1, sim.timestep, total_steps)
        updater = hoomd.update.BoxResize(trigger, box_i, box_f, variant)
        sim.operations.updaters.append(updater)

        write_trigger = hoomd.trigger.Periodic(1_000, phase=sim.timestep)
        tmp_file = output_dir + "/_analysis_traj.gsd"
        traj = output_dir + "/analysis_traj.gsd"
        writer = hoomd.write.GSD(filename=tmp_file, trigger=write_trigger, mode="wb", log=log)
        sim.operations.writers.append(writer)

        # print(total_steps)

        sim.run(total_steps)

        del sim, box_i, box_f, log, analyzer, xy_logger, updater, writer, trigger, variant, write_trigger

        gc.collect()

        shutil.move(tmp_file, traj)

    runs = sorted(glob.glob(job.fn("shear_runs/rate-*/temp-*/traj.gsd")))
    processes = []
    # print(runs)
    for run in runs:
        temp = utils.extract_between(run, "temp-", "/traj.gsd")
        shear_rate = float(utils.extract_between(run, "rate-", "/temp-"))
        print("temp:", temp)
        max_strain = total_steps*shear_rate*dt
        process = multiprocessing.Process(target=job_task, args=(job, max_strain, shear_rate, temp))
        process.start()
        # processes.append(process)
        process.join()

    job.doc["const_shear_anaysis_ran"] = True


if __name__ == "__main__":
    Project.get_project(root=config["root"]).main()