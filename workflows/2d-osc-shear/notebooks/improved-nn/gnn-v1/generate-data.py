import numpy as np
import polars as pl
import gsd.hoomd
import schmeud._schmeud as schmeud_rs
from schmeud import ml
from tqdm import tqdm
import argparse

import glob
import os
import pathlib
import pickle
import signac
import freud
from numba import njit

from dataclasses import dataclass
from collections import defaultdict

import matplotlib.pyplot as plt
from scipy import stats

from monk import workflow, utils

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

args = argparser.parse_args()
DRY_RUN = args.dry_run

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

for job in project:
    print(job)
    prep = job.sp["prep"]
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

    pbar = tqdm(len(experiments)*18, desc="Experiments", postfix={"skipped": 0})
    for exper in experiments:
        max_shear = utils.extract_between(exper, "max-shear-", "/")
        period = utils.extract_between(exper, "period-", ".gsd")
        temp = utils.extract_between(exper, "temp-", "/")

        traj = gsd.hoomd.open(exper)

        cycle_start_idx = lambda i: -1 + i * 40

        for idx in np.arange(1, 19):
            idx = int(idx)
            frame = cycle_start_idx(idx)
            
            df_path = id_path / f"dataset_max-shear-{max_shear}_period-{period}_temp-{temp}_frame-{frame}.npz"

            if os.path.exists(df_path.as_posix()):
                # print(f"Skipping {df_path}")
                skipped += 1
                pbar.set_postfix({"skipped": skipped})
                pbar.update(1)
                continue

            snap = traj[frame]

            typeids = snap.particles.typeid

            # softness of first rev frame
            snap_high = traj[10 + frame]

            pos = snap.particles.position
            pos_high = snap_high.particles.position
 
            frames = np.array([frame])

            strains = np.array([max_shear])

            box = snap.configuration.box[:]
            box_high = snap_high.configuration.box[:]

            nlist_query = freud.locality.LinkCell.from_system(snap)
            nlist_20 = nlist_query.query(
                snap.particles.position, {"num_neighbors": 20, "exclude_ii": True}
            ).toNeighborList()

            freud_box = freud.box.Box.from_box(box)
            disp_20 = freud_box.wrap(pos[nlist_20.query_point_indices] - pos[nlist_20.point_indices])

            nlist_query_high = freud.locality.LinkCell.from_system(snap_high)
            nlist_high_20 = nlist_query.query(
                snap_high.particles.position, {"num_neighbors": 20, "exclude_ii": True}
            ).toNeighborList()

            freud_box_high = freud.box.Box.from_box(box_high)
            disp_high_20 = freud_box_high.wrap(pos_high[nlist_high_20.query_point_indices] - pos_high[nlist_high_20.point_indices])

            dataset = {
                "frame": frames,
                "strain": strains,
                "id": np.array(typeids),
                "pos": np.array(snap.particles.position),
                "box": np.array(box),
                "qidx_20": np.array(nlist_20.query_point_indices),
                "tidx_20": np.array(nlist_20.point_indices),
                "disp_20": np.array(disp_20),
                "pos_high": np.array(snap_high.particles.position),
                "box_high": np.array(box_high),
                "qidx_high_20": np.array(nlist_high_20.query_point_indices),
                "tidx_high_20": np.array(nlist_high_20.point_indices),
                "disp_high_20": np.array(disp_high_20),

            }
            np.savez(df_path.as_posix(), **dataset)
            pbar.update(1)

        # print size of data in folder
        # print(f"Size of data in folder: {get_dir_size(id_path.as_posix())}")
