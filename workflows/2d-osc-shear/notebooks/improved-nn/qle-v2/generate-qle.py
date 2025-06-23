import numpy as np
import polars as pl
import gsd.hoomd
import schmeud._schmeud as schmeud_rs
from schmeud import ml
from tqdm import tqdm
import argparse
import hoomd

import glob
import os
import pathlib
import pickle
import signac
import freud
from numba import njit
import sys

from dataclasses import dataclass
from collections import defaultdict

import matplotlib.pyplot as plt
from scipy import stats

from monk import workflow, utils, pair, prep

import scipy.stats as stats
import multiprocessing as mp

import qle

K = 1024 + 3
USE_GPU = True

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
    
    default_fire_kwargs = {"dt": 1e-2, "force_tol": 1e-10, "angmom_tol": 1.0, "energy_tol": 1e-10}
    
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
        nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All())
    else:
        nve = method
    fire.forces = [force]
    fire.methods = [nve]

    sim.operations.integrator = fire

    return fire, nve

def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

argparser = argparse.ArgumentParser()

argparser.add_argument("--dry-run", action="store_true")
argparser.add_argument("--jobs", nargs="+", type=str)

args = argparser.parse_args()
DRY_RUN = args.dry_run
JOBS = args.jobs

cwd = pathlib.Path(os.getcwd())
parent = cwd.parent.parent.parent / "config.yaml"
config = workflow.get_config(parent.as_posix())
local = cwd / "config.yaml"
local_config = workflow.get_any_yaml(local.as_posix())

output_folder = pathlib.Path(local_config["data"])

project: signac.Project = signac.get_project(root=config["root"])


if DRY_RUN:
    print("Dry run")
    print(f"Output folder: {output_folder}")

if USE_GPU:
    device = hoomd.device.GPU()
else:
    device = hoomd.device.CPU()

for job in project:
    if JOBS is not None and job.id not in JOBS:
        print(f"Skipping job, {job.id} is not in {JOBS}")
        continue
    print(job)
    # jprep = job.sp["prep"]
    job_id = job.id
    experiments = sorted(glob.glob(job.fn("experiments/*/*/traj-fire_period-*.gsd")))
    # add aqs_therm_data
    experiments += sorted(glob.glob(job.fn("aqs_therm_experiments/*/*/traj-fire_period-*.gsd")))
    if len(experiments) == 0:
        continue

    id_path = output_folder / f"{job_id}"

    id_path.mkdir(parents=True, exist_ok=True)

    print(f"Processing {job_id}")
    print(f"Saving to {id_path}")
    skipped = 0

    if DRY_RUN:
        continue

    sim = hoomd.Simulation(device=device)

    sim_initilized = False

    postfix = {"skipped": 0}
    pbar = tqdm(desc="Experiments", postfix=postfix)
    for exper in experiments:

        max_shear = utils.extract_between(exper, "max-shear-", "/")
        period = utils.extract_between(exper, "period-", ".gsd")
        temp = utils.extract_between(exper, "temp-", "/")

        if temp not in ["1.9836e-03", "0.0496", "0.0992"]:
            continue

        # print(temp)

        traj = gsd.hoomd.open(exper)

        if not sim_initilized:
            sim_initilized = True
            sim.create_state_from_snapshot(traj[39])
            fire, nve = _setup_fire_sim(job, sim)
            postfix["step"] = "A"
            pbar.set_postfix(postfix)
            sim.run(0)
            postfix["step"] = "B"
            pbar.set_postfix(postfix)


        cycle_start_idx = lambda i: -1 + i * 40

        for idx in np.arange(1, 19, 4):
            try:
                idx = int(idx)
                frame = cycle_start_idx(idx)
                postfix["step"] = "C"
                pbar.set_postfix(postfix)
                hoomd_snap = hoomd.Snapshot.from_gsd_frame(traj[frame], device.communicator)
                sim.state.set_snapshot(hoomd_snap)
                postfix["step"] = "D"

                # print("snapshot set")

                df_path = id_path / f"dataset_max-shear-{max_shear}_period-{period}_temp-{temp}_frame-{frame}.npz"

                if df_path.exists():
                    skipped += 1
                    # print("skipped!")
                    postfix["skipped"] = skipped
                    pbar.set_postfix(postfix)
                    pbar.update(1)
                    continue

                postfix["step"] = "E"
                pbar.set_postfix(postfix)
                
                # print("hello")
                fire.reset()

                postfix["converged"] = False
                while not fire.converged:
                    sim.run(1000)
                    postfix["max_force"] = np.max(fire.forces[0].forces)
                    pbar.set_postfix(postfix)
                postfix["converged"] = True
                pbar.set_postfix(postfix)

                snap = sim.state.get_snapshot()
                N = len(snap.particles.position)

                computer = qle.QLM()

                # print("Computing")

                evals, evecs, hessian, hj, half_edges = computer.compute(snap, dense=False, k=K, dense_filter=False, filter=True, gpu=USE_GPU, dont_apply_filter=True)
                evecs = np.ascontiguousarray(evecs.T)

                filtered_evecs = []

                # argidx to get first value above 1e-10
                toss_idx = np.argmax(evals > 1e-4)

                print(toss_idx, evals[:10])

                # print(evals[:10])

                # print("computing vibrality")

                evecs_rshp = evecs.reshape((evecs.shape[0], N, 2))

                v_16 = qle.compute_output(evals, evecs_rshp, modes=16, toss=toss_idx)
                v_32 = qle.compute_output(evals, evecs_rshp, modes=32, toss=toss_idx)
                v_64 = qle.compute_output(evals, evecs_rshp, modes=64, toss=toss_idx)
                v_128 = qle.compute_output(evals, evecs_rshp, modes=128, toss=toss_idx)
                v_256 = qle.compute_output(evals, evecs_rshp, modes=256, toss=toss_idx)
                v_512 = qle.compute_output(evals, evecs_rshp, modes=512, toss=toss_idx)
                v_1024 = qle.compute_output(evals, evecs_rshp, modes=1024, toss=toss_idx)

                # print("computing output")

                sema = mp.BoundedSemaphore(mp.cpu_count())
                manager = mp.Manager()
                return_dict = manager.dict()

                def worker(sema, procid, return_dict, half_edges, orig_vs, hj, dim, N):
                    try:
                        outs = []
                        for v in orig_vs:
                            norm = np.linalg.norm(v)
                            v = v / norm
                            out = computer.filter_vec_sparse(half_edges, v, hj, dim, N)
                            outs.append(out)
                        return_dict[procid] = outs
                    finally:
                        sema.release()
                    
                procs = []

                chunk_size = 10
                grid = int(np.ceil(K / chunk_size))

                for idx in tqdm(range(grid), desc="filtering eigenvectors"):

                    orig_vs = evecs[idx*chunk_size:(idx+1)*chunk_size]

                    sema.acquire()
                    procs.append(mp.Process(target=worker, args=(sema, idx, return_dict, half_edges, orig_vs, hj, 2, N)))
                    procs[-1].start()

                    while not procs[0].is_alive():
                        procs.pop(0)

                for p in procs:
                    p.join()

                for i in range(grid):
                    filtered_evecs.extend(return_dict[i])

                # print("computing filtered vibrality")

                evecs = np.array(filtered_evecs)

                evecs_rshp = evecs.reshape((evecs.shape[0], N, 2))

                f_16 = qle.compute_output(evals, evecs_rshp, modes=16, toss=toss_idx)
                print(f_16)
                f_32 = qle.compute_output(evals, evecs_rshp, modes=32, toss=toss_idx)
                print(f_32)
                f_64 = qle.compute_output(evals, evecs_rshp, modes=64, toss=toss_idx)
                f_128 = qle.compute_output(evals, evecs_rshp, modes=128, toss=toss_idx)
                f_256 = qle.compute_output(evals, evecs_rshp, modes=256, toss=toss_idx)
                f_512 = qle.compute_output(evals, evecs_rshp, modes=512, toss=toss_idx)
                f_1024 = qle.compute_output(evals, evecs_rshp, modes=1024, toss=toss_idx)


                # print("done")

                # print("correlation")

                dataset = {
                    "pos": snap.particles.position,
                    "box": snap.configuration.box,
                    "v16": v_16,
                    "v32": v_32,
                    "v64": v_64,
                    "v128": v_128,
                    "v256": v_256,
                    "v512": v_512,
                    "v1024": v_1024,
                    "f16": f_16,
                    "f32": f_32,
                    "f64": f_64,
                    "f128": f_128,
                    "f256": f_256,
                    "f512": f_512,
                    "f1024": f_1024,
                }
                np.savez(df_path.as_posix(), **dataset)
                print(f"Saved to {df_path.as_posix()}")
                pbar.update(1)
                # sys.exit()
            except Exception as e:
                print(e)
                pbar.update(1)
                continue

