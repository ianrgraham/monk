#!python

from typing import Tuple
import flow
import signac
import hoomd
import gsd.hoomd
import freud
import numpy as np
from schmeud.dynamics import thermal
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


# Custom hoomd triggers and actions

class AsyncTrigger(hoomd.trigger.Trigger):
    """NOT async in the Rust or JavaScript way
    
    Trigger by calling the activate() method"""

    def __init__(self):
        self.async_trig = False
        hoomd.trigger.Trigger.__init__(self)

    def activate(self):
        self.async_trig = True

    def compute(self, timestep):
        out = self.async_trig
        self.async_trig = False
        return out

class RestartablePeriodicTrigger(hoomd.trigger.Trigger):

    def __init__(self, period):
        assert(period >= 1)
        self.period = period
        self.state = period - 1
        hoomd.trigger.Trigger.__init__(self)

    def reset(self):
        self.state = self.period - 1

    def compute(self, timestep):
        if self.state >= self.period - 1:
            # print("Triggering", timestep, self.state, self.period)
            self.state = 0
            return True
        else:
            self.state += 1
            return False


class UpdatePos(hoomd.custom.Action):
    """Custom action to handle feeding in a new set of positions through a snapshot"""

    def __init__(self, new_snap=None):
        self.new_snap = new_snap

    def set_snap(self, new_snap):
        self.new_snap = new_snap

    def act(self, timestep):
        # print("Updating positions", timestep)
        old_snap = self._state.get_snapshot()
        if old_snap.communicator.rank == 0:
            N = old_snap.particles.N
            new_velocity = np.zeros((N,3))
            for i in range(N):
                old_snap.particles.velocity[i] = new_velocity[i]
                old_snap.particles.position[i] = self.new_snap.particles.position[i]
        self._state.set_snapshot(old_snap)

class PastSnapshotsBuffer(hoomd.custom.Action):
    """Custom action to hold onto past simulation snapshots"""

    def __init__(self):
        self.snap_buffer = []

    def clear(self):
        self.snap_buffer.clear()

    def get_snapshots(self):
        return self.snap_buffer

    def force_push(self):
        self.act(None)

    def act(self, timestep):
        # print("Pushing snapshot", timestep)
        snap = self._state.get_snapshot()
        self.snap_buffer.append(snap)



@Project.operation
@Project.pre.true('softness_trained')
@Project.pre(lambda job: job.sp["replica"] == 0)
@Project.post.true('isconfig_fire_ran')
def isoconfig_fire_runs(job: signac.Project.Job):

    print("Job ID:", job.id)

    replicas = 100
    total_steps = 2000
    runs = sorted(glob.glob(job.fn("short_runs/*/fire_traj.gsd")))[:10]

    print(runs)

    sp = job.sp
    doc = job.doc

    delta = sp["delta"]

    sim = hoomd.Simulation(hoomd.device.GPU(), seed=doc["seed"])
    sim.create_state_from_gsd(job.fn("init.gsd"))

    for run in runs:

        output_file = run.replace("fire_traj.gsd", "fire_isconfig.parquet")
        assert output_file != run

        if os.path.exists(output_file):
            print("Skipping", output_file)
            continue
        else:
            print(run)

        temp = float(utils.extract_between(run, "temp-", "/"))
        traj = gsd.hoomd.open(run)

        print(temp)

        integrator = hoomd.md.Integrator(dt=0.005)
        nlist = hoomd.md.nlist.Tree(0.3)
        mlj = pair.KA_MLJ(nlist, delta)
        integrator.forces.append(mlj)

        nvt = hoomd.md.methods.NVT(
            kT=temp,
            filter=hoomd.filter.All(),
            tau=0.5)
        integrator.methods.append(nvt)
        sim.operations.integrator = integrator

        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=temp)

        all_phops = []

        for jdx in range(len(traj))[::10]:
            print("jdx:", jdx)
            snap = traj[jdx]

            custom_updater = UpdatePos(new_snap=snap)
            snap_buffer = PastSnapshotsBuffer()
            reset_config_trig = AsyncTrigger()
            snap_buffer_trig = RestartablePeriodicTrigger(200)

            custom_op = hoomd.update.CustomUpdater(action=custom_updater,
                                                trigger=reset_config_trig)

            another_op = hoomd.update.CustomUpdater(action=snap_buffer,
                                                    trigger=snap_buffer_trig)
            
            sim.operations.updaters.clear()
            sim.operations.updaters.append(custom_op)
            sim.operations.updaters.append(another_op)


            for idx in range(replicas):
                # print("timestep", sim.timestep)
                reset_config_trig.activate()
                snap_buffer_trig.reset()
                sim.run(2)
                sim.state.thermalize_particle_momenta(hoomd.filter.All(), temp)

                sim.run(total_steps+2)

                # process phop
                mtraj = snap_buffer.get_snapshots()
                phop = thermal.calc_phop(mtraj, tr_frames=len(mtraj)-1)

                box = freud.box.Box.from_box(snap.configuration.box)

                prop = np.linalg.norm(box.wrap(mtraj[-1].particles.position - mtraj[0].particles.position), axis=1)

                all_phops.append([np.uint32(jdx), np.arange(len(phop[0]), dtype=np.uint32), np.uint32(idx), phop[0].astype(np.float32), prop.astype(np.float32)]) # trivial unwrapping

                snap_buffer.clear()

        df = pd.DataFrame(all_phops, columns=["frame", "id", "replica", "phop", "prop"]).explode(["id", "phop", "prop"])

        df.to_parquet(output_file)

    doc["isconfig_fire_ran"] = True



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


@Project.operation
@Project.pre.true('analysis_sfs_computed')
@Project.post.true('struct_descrs_computed')
def compute_struct_descr(job: signac.Project.Job):
    """Compute structural descriptors for analysis."""

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
        df_soft = []


        print("Computing structure functions")
        for frame in range(0, len(phop), 2):
            # soft_indices = np.array(soft_indices, dtype=np.uint32)
            # hard_indices = np.array(hard_indices, dtype=np.uint32)
            snap: gsd.hoomd.Snapshot = traj[int(frame)]
            num_particles = snap.particles.N
            a_type = snap.particles.typeid == 0
            # query_indices = np.concatenate([soft_indices, hard_indices])
            # perm = query_indices.argsort()
            # sort arrays by query indices

            # get sfs
            sfs = ml.compute_structure_functions_snap(snap)[a_type]

            df_frame.append(frame*np.ones(num_particles, dtype=np.uint32))
            df_tag.append(np.arange(num_particles)[a_type])
            df_type.append(snap.particles.typeid)
            df_phop.append(phop[frame])
            # df_sf.append(sfs)

        df = pd.DataFrame(
            {
                "frame": np.concatenate(df_frame),
                "tag": np.concatenate(df_tag),
                "type": np.concatenate(df_type),
                "phop": np.concatenate(df_phop),
                "soft": np.concatenate(df_soft),
            }
        )

        out_file = run.replace("fire_traj.gsd", "sig_struct_descrs.parquet")
        assert out_file != run

        df.to_parquet(out_file)

    job.doc["struct_descrs_computed"] = True


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

    # shear_rates = [1e-3, 1e-4, 1e-5]
    shear_rates = [2e-3, 5e-4, 2e-4]
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
@Project.post.true("const_shear_anaysis_ran")  # FIX ME: typo "analysis"
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
        tmp_file = output_dir + "/_analysis_traj.gsd"
        traj = output_dir + "/analysis_traj.gsd"
        if os.path.exists(traj):
            return
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
        print("box_xy:", box_i.xy, box_f.xy)
        
        total_steps = int(np.round(max_strain/shear_rate/dt))
        shear_update_period = 1 # int(np.round(step_size/shear_rate))
        trigger = hoomd.trigger.Periodic(shear_update_period, phase=sim.timestep)
        variant = hoomd.variant.Ramp(0, 1, sim.timestep, total_steps)
        updater = hoomd.update.BoxResize(trigger, box_i, box_f, variant)
        sim.operations.updaters.append(updater)

        write_trigger = hoomd.trigger.Periodic(1_000, phase=sim.timestep)
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
        print("temp:", temp, "shear_rate:", shear_rate)
        max_strain = total_steps*shear_rate*dt
        process = multiprocessing.Process(target=job_task, args=(job, max_strain, shear_rate, temp))
        process.start()
        # processes.append(process)
        process.join()

    job.doc["const_shear_anaysis_ran"] = True


@Project.operation
@Project.pre.true('const_shear_anaysis_ran')
@Project.post.true('shear_fire_applied')
def fire_minimize_shear_analysis_runs(job: signac.Project.Job):
    """Apply FIRE minimization to the shear runs ready for analysis."""
    sp = job.sp
    doc = job.doc

    print(sp, doc)

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
    runs = glob.glob(job.fn("shear_runs/*/*/analysis_traj.gsd"))
    print(len(runs))
    for run in runs:
        print(run)
        traj = gsd.hoomd.open(run)
        fire_traj = run.replace("analysis_traj.gsd", "fire_traj.gsd")

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

    job.doc["shear_fire_applied"] = True


@Project.operation
@Project.pre.true('shear_fire_applied')
@Project.post.true('shear_analysis_sfs_computed')
def compute_shear_analysis_sfs(job: signac.Project.Job):
    """Compute stucture functions for sheared systems analysis."""

    print("Job ID:", job.id)
    # only one run should be used
    runs = glob.glob(job.fn("shear_runs/*/*/fire_traj.gsd"))
    for run in runs:
        print(run)
        out_file = run.replace("fire_traj.gsd", "sfs.parquet")
        assert out_file != run
        if os.path.isfile(out_file):
            continue
        traj = gsd.hoomd.open(run)
        pos = []
        box0 = freud.box.Box.from_box(traj[0].configuration.box)
        pos0 = box0.make_fractional(traj[0].particles.position)
        unit_box = freud.box.Box.cube(1)
        prev = pos0
        image = np.zeros((len(traj[0].particles.position), 3), dtype=np.int32)
        for frame in traj:
            box = freud.box.Box.from_box(frame.configuration.box)
            new = box.make_fractional(frame.particles.position)
            image -= unit_box.get_images(new - prev)
            x = box.make_absolute(new)
            x = box.unwrap(x, image)
            x -= box.make_absolute(pos0)
            pos.append(x)
            prev = new
        pos = np.array(pos)

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

        df.to_parquet(out_file)

    job.doc["shear_analysis_sfs_computed"] = True

# stress simulations
@Project.operation
@Project.pre(lambda job: job.sp["delta"] == 0.0 and job.sp["replica"] == 0)
@Project.post.true('const_stress_ran')
def constant_stress_runs(job: signac.Project.Job):
    """Run constant stress simulations."""
    sp = job.sp
    doc = job.doc

    print(job, sp, doc)

    project = job._project

    dt = project.doc["dt"]

    # stresses = [0.02, 0.05, 0.1]

    stresses = [0.2, 0.3, 0.4]

    total_steps = 1_000_000

    def setup_sim(gsd_file, temp, stress):
        sim = hoomd.Simulation(hoomd.device.GPU(), seed=doc["seed"])
        sim.create_state_from_gsd(gsd_file)

        integrator = hoomd.md.Integrator(dt)

        nlist = hoomd.md.nlist.Tree(buffer=0.3)
        mlj = pair.KA_ModLJ(nlist, sp["delta"])
        nve = hoomd.md.methods.NPT(hoomd.filter.All(), temp, 0.1, [0, 0, 0, 0, 0, stress], 0.1, couple="none", box_dof=[False, False, False, True, False, False])
        integrator.forces = [mlj]
        integrator.methods = [nve]
        sim.operations.integrator = integrator

        return sim

    def job_task(job, run, stress, temp):
        output_dir = job.fn(f"stress_runs/stress-{stress}/temp-{temp}")
        if os.path.exists(output_dir + "/traj.gsd"):
            return
        os.makedirs(output_dir, exist_ok=True)

        sim = setup_sim(run, float(temp), stress)

        sim.always_compute_pressure = True

        log = hoomd.logging.Logger()

        analyzer = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
        xy_logger = xyLogger(sim)
        sim.operations.computes.append(analyzer)

        log.add(analyzer, quantities=["pressure", "pressure_tensor", "kinetic_temperature"])
        log.add(sim, quantities=["timestep"])
        log[('box', 'xy')] = (xy_logger, 'value', 'scalar')

        write_trigger = hoomd.trigger.Periodic(10_000, phase=sim.timestep)
        tmp_file = output_dir + "/_traj.gsd"
        traj = output_dir + "/traj.gsd"
        writer = hoomd.write.GSD(filename=tmp_file, trigger=write_trigger, mode="wb", log=log)
        sim.operations.writers.append(writer)

        # print(total_steps)

        sim.run(total_steps)

        del sim, log, analyzer, xy_logger, writer, write_trigger

        gc.collect()

        shutil.move(tmp_file, traj)
    
    runs = sorted(glob.glob(job.fn("runs/temp-*/traj.gsd")))
    # print(runs)
    for run in runs[:10:2]:
        temp = utils.extract_between(run, "temp-", "/traj.gsd")
        print("temp:", temp)
        # processes = []
        for stress in stresses:
            process = multiprocessing.Process(target=job_task, args=(job, run, stress, temp))
            process.start()
            process.join()
        #     processes.append(process)
        # for process in processes:
        #     process.join()

    job.doc["const_stress_ran"] = True

@Project.operation
@Project.pre.true("const_stress_ran")
@Project.post.true("const_stress_analysis_ran")
def constant_stress_runs_analysis(job: signac.Project.Job):
    """Run constant stress simulations."""
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
        nve = hoomd.md.methods.NPT(hoomd.filter.All(), temp, 0.1, [0, 0, 0, 0, 0, stress], 0.1, couple="none", box_dof=[False, False, False, True, False, False])
        integrator.forces = [mlj]
        integrator.methods = [nve]
        sim.operations.integrator = integrator

        sim.run(0)

        return sim

    def job_task(job, run, stress, temp):
        output_dir = job.fn(f"stress_runs/stress-{stress}/temp-{temp}")
        tmp_file = output_dir + "/_analysis_traj.gsd"
        traj = output_dir + "/analysis_traj.gsd"
        if os.path.exists(traj): # skip if already done
            return
        sim = setup_sim(run, float(temp))

        sim.always_compute_pressure = True

        log = hoomd.logging.Logger()

        analyzer = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
        xy_logger = xyLogger(sim)
        sim.operations.computes.append(analyzer)

        log.add(analyzer, quantities=["pressure", "pressure_tensor", "kinetic_temperature"])
        log.add(sim, quantities=["timestep"])
        log[('box', 'xy')] = (xy_logger, 'value', 'scalar')

        # box_f = sim.state.box
        # while box_f.xy < -0.5 and box_f.xy > -1.0:
        #     print("adjust box")
        #     box_i = sim.state.box
        #     box_f = copy.deepcopy(box_i)
        #     box_f.xy = box_i.xy + 1.0
        #     new_xy = box_f.xy
        #     print("box_xy:", box_i.xy, box_f.xy)
        #     print(box_f)
        #     hoomd.update.BoxResize.update(sim.state, box_f, hoomd.filter.All())

        #     print(sim.state.box.xy, "==", new_xy)
        #     assert sim.state.box.xy == new_xy
        # return

        write_trigger = hoomd.trigger.Periodic(1_000, phase=sim.timestep)
        
        # if os.path.exists(traj):
        #     return
        writer = hoomd.write.GSD(filename=tmp_file, trigger=write_trigger, mode="wb", log=log)
        sim.operations.writers.append(writer)

        # print(total_steps)

        sim.run(total_steps)

        del sim, log, analyzer, xy_logger, writer, write_trigger

        gc.collect()

        shutil.move(tmp_file, traj)

    runs = sorted(glob.glob(job.fn("stress_runs/stress-*/temp-*/traj.gsd")))
    # print(runs)
    for run in runs:
        temp = utils.extract_between(run, "temp-", "/traj.gsd")
        stress = float(utils.extract_between(run, "stress-", "/temp-"))
        print("temp:", temp, "stress:", stress)
        process = multiprocessing.Process(target=job_task, args=(job, run, stress, temp))
        process.start()
        process.join()

    job.doc["const_stress_analysis_ran"] = True


@Project.operation
@Project.pre.true('const_stress_analysis_ran')
@Project.post.true('stress_fire_applied')
def fire_minimize_stress_analysis_runs(job: signac.Project.Job):
    """Apply FIRE minimization to the stress runs ready for analysis."""
    sp = job.sp
    doc = job.doc

    print(sp, doc)

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
    runs = glob.glob(job.fn("stress_runs/*/*/analysis_traj.gsd"))
    print(len(runs))
    for run in runs:
        print(run)
        traj = gsd.hoomd.open(run)
        fire_traj = run.replace("analysis_traj.gsd", "fire_traj.gsd")

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

    job.doc["stress_fire_applied"] = True


@Project.operation
@Project.pre.true('stress_fire_applied')
@Project.post.true('stress_analysis_sfs_computed')
def compute_stress_analysis_sfs(job: signac.Project.Job):
    """Compute stucture functions for stressed systems analysis."""

    print("Job ID:", job.id)
    # only one run should be used
    runs = glob.glob(job.fn("stress_runs/*/*/fire_traj.gsd"))
    for run in runs:
        print(run)
        out_file = run.replace("fire_traj.gsd", "sfs.parquet")
        assert out_file != run
        if os.path.isfile(out_file):
            continue
        traj = gsd.hoomd.open(run)
        pos = []
        box0 = freud.box.Box.from_box(traj[0].configuration.box)
        pos0 = box0.make_fractional(traj[0].particles.position)
        unit_box = freud.box.Box.cube(1)
        prev = pos0
        image = np.zeros((len(traj[0].particles.position), 3), dtype=np.int32)
        for frame in traj:
            box = freud.box.Box.from_box(frame.configuration.box)
            new = box.make_fractional(frame.particles.position)
            image -= unit_box.get_images(new - prev)
            x = box.make_absolute(new)
            x = box.unwrap(x, image)
            x -= box.make_absolute(pos0)
            pos.append(x)
            prev = new
        pos = np.array(pos)

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

        df.to_parquet(out_file)

    job.doc["stress_analysis_sfs_computed"] = True

if __name__ == "__main__":
    Project.get_project(root=config["root"]).main()