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
import hoomd
from monk import pair


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

pipe = None
pipe0 = None
pipe1 = None
with open("../../sklearn/svc.pkl", "rb") as f:
    pipe = pickle.load(f)

with open("../../sklearn/svc_type0.pkl", "rb") as f:
    pipe0 = pickle.load(f)

with open("../../sklearn/svc_type1.pkl", "rb") as f:
    pipe1 = pickle.load(f)


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

        # if (
        #     float(period) != 1000.0
        #     or float(temp) != 0.019836
        #     or float(max_shear) > 0.04
        # ):
        #     continue

        traj = gsd.hoomd.open(exper)

        cycle_start_idx = lambda i: -1 + i * 40

        for idx in np.arange(1, 19):
            idx = int(idx)
            frame = cycle_start_idx(idx)
            
            df_path = id_path / f"dataset_max-shear-{max_shear}_period-{period}_temp-{temp}_frame-{frame}.parquet"

            if os.path.exists(job.fn(df_path)):
                # print(f"Skipping {df_path}")
                skipped += 1
                pbar.set_postfix({"skipped": skipped})
                pbar.update(1)
                continue

            snap = traj[frame]

            typeids = snap.particles.typeid
            # softness of first frame
            softb = np.zeros_like(typeids, dtype=np.float32)

            query_indices0 = np.arange(snap.particles.N)[typeids == 0]
            sfs0 = ml.compute_structure_functions_snap(snap, query_indices0)
            soft0 = pipe0.decision_function(sfs0)

            query_indices1 = np.arange(snap.particles.N)[typeids == 1]
            sfs1 = ml.compute_structure_functions_snap(snap, query_indices1)
            soft1 = pipe1.decision_function(sfs1)

            softb[typeids == 0] = soft0
            softb[typeids == 1] = soft1

            all_sfs = np.zeros((snap.particles.N, sfs1.shape[1]), dtype=np.float32)

            all_sfs[typeids == 0] = sfs0
            all_sfs[typeids == 1] = sfs1

            # softness of first rev frame
            snap_high = traj[10 + frame]
            softr = np.zeros_like(typeids, dtype=np.float32)

            query_indices0 = np.arange(snap_high.particles.N)[typeids == 0]
            sfs0 = ml.compute_structure_functions_snap(snap_high, query_indices0)
            soft0 = pipe0.decision_function(sfs0)

            query_indices1 = np.arange(snap_high.particles.N)[typeids == 1]
            sfs1 = ml.compute_structure_functions_snap(snap_high, query_indices1)
            soft1 = pipe1.decision_function(sfs1)

            softr[typeids == 0] = soft0
            softr[typeids == 1] = soft1

            all_sfsr = np.zeros((snap_high.particles.N, sfs1.shape[1]), dtype=np.float32)

            all_sfsr[typeids == 0] = sfs0
            all_sfsr[typeids == 1] = sfs1

            frames = np.ones_like(typeids, dtype=np.int32) * frame

            strains = np.ones_like(typeids, dtype=np.float32) * float(max_shear)

            snap_low = traj[30 + frame]

            snap_next = traj[40 + frame]

            box = snap.configuration.box[:]
            box_high = snap_high.configuration.box[:]
            box_low = snap_low.configuration.box[:]
            box_next = snap_next.configuration.box[:]

            nlist_query = freud.locality.LinkCell.from_system(snap)
            nlist = nlist_query.query(
                snap.particles.position, {"num_neighbors": 20, "exclude_ii": True}
            ).toNeighborList()

            d2min_1 = schmeud_rs.dynamics.d2min_frame(
                snap.particles.position[:, :2],
                snap_low.particles.position[:, :2],
                nlist.query_point_indices,
                nlist.point_indices,
                (box, box_low),
            )

            d2min_2 = schmeud_rs.dynamics.d2min_frame(
                snap.particles.position[:, :2],
                snap_high.particles.position[:, :2],
                nlist.query_point_indices,
                nlist.point_indices,
                (box, box_high),
            )

            d2min_irr = schmeud_rs.dynamics.d2min_frame(
                snap.particles.position[:, :2],
                snap_next.particles.position[:, :2],
                nlist.query_point_indices,
                nlist.point_indices,
                (box, box_next),
            )

            nlist_query = freud.locality.LinkCell.from_system(snap_high)
            nlist = nlist_query.query(
                snap_high.particles.position, {"num_neighbors": 20, "exclude_ii": True}
            ).toNeighborList()

            d2min_rev = schmeud_rs.dynamics.d2min_frame(
                snap_high.particles.position[:, :2],
                snap_low.particles.position[:, :2],
                nlist.query_point_indices,
                nlist.point_indices,
                (box_high, box_low),
            )

            dataset = pl.DataFrame(
                {
                    "frame": frames,
                    "strain": strains,
                    "id": np.array(typeids),
                    "soft": np.array(softb),
                    "sfs": all_sfs,
                    "soft_rev": np.array(softr),
                    "sfs_rev": all_sfsr,
                    "d2min_rev": d2min_rev,
                    "d2min_left": d2min_1,
                    "d2min_right": d2min_2,
                    "d2min_irr": d2min_irr,
                }
            )
            dataset.write_parquet(job.fn(df_path), use_pyarrow=True)
            pbar.update(1)

        # print size of data in folder
        # print(f"Size of data in folder: {get_dir_size(id_path.as_posix())}")
