import numpy as np
import polars as pl
import gsd.hoomd
import schmeud._schmeud as schmeud_rs
from schmeud import ml
from tqdm import tqdm

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

parent = pathlib.Path(os.getcwd()).parent.parent / "config.yaml"
config = workflow.get_config(parent.as_posix())

project: signac.Project = signac.get_project(root=config["root"])

pipe = None
pipe0 = None
pipe1 = None
with open("../sklearn/svc.pkl", "rb") as f:
    pipe = pickle.load(f)

with open("../sklearn/svc_type0.pkl", "rb") as f:
    pipe0 = pickle.load(f)

with open("../sklearn/svc_type1.pkl", "rb") as f:
    pipe1 = pickle.load(f)


for job in project:
    print(job)
    prep = job.sp["prep"]

    experiments = sorted(glob.glob(job.fn("longer_experiments/*/*/traj-fire_period-*.gsd")))
    if len(experiments) == 0:
        continue

    for exper in tqdm(experiments):
        max_shear = utils.extract_between(exper, "max-shear-", "/")
        period = utils.extract_between(exper, "period-", ".gsd")
        temp = utils.extract_between(exper, "temp-", "/")

        if (
            float(period) != 1000.0
            or float(temp) != 0.019836
            or float(max_shear) > 0.04
        ):
            continue

        traj = gsd.hoomd.open(exper)

        cycle_start_idx = lambda i: -1 + i * 40

        for idx in [0, 4, 8, 12, 16, 20]:
            frame = cycle_start_idx(idx)
            df_path = f"longer_experiments/max-shear-{max_shear}/temp-{temp}/vae-dataset_period-{period}_frame-{frame}.parquet"

            if os.path.exists(job.fn(df_path)):
                print(f"Skipping {df_path}")
                continue

            snap = traj[frame]

            typeids = snap.particles.typeid
            softb = np.zeros_like(typeids, dtype=np.float32)

            query_indices0 = np.arange(snap.particles.N)[typeids == 0]
            sfs0 = ml.compute_structure_functions_snap(snap, query_indices0)
            soft0 = pipe0.decision_function(sfs0)

            query_indices1 = np.arange(snap.particles.N)[typeids == 1]
            sfs1 = ml.compute_structure_functions_snap(snap, query_indices1)
            soft1 = pipe1.decision_function(sfs1)

            softb[typeids == 0] = soft0
            softb[typeids == 1] = soft1

            # softness.append(softb)

            all_sfs = np.zeros((snap.particles.N, sfs1.shape[1]), dtype=np.float32)

            all_sfs[typeids == 0] = sfs0
            all_sfs[typeids == 1] = sfs1

            frames = np.ones_like(typeids, dtype=np.int32) * frame

            strains = np.ones_like(typeids, dtype=np.float32) * float(max_shear)

            snap_high = traj[10 + frame]
            snap_low = traj[30 + frame]

            box = snap.configuration.box[:]
            box_high = snap_high.configuration.box[:]
            box_low = snap_low.configuration.box[:]

            nlist_query = freud.locality.LinkCell.from_system(snap)
            nlist = nlist_query.query(
                snap.particles.position, {"num_neighbors": 10}
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
                    "d2min_rev_10": d2min_rev,
                    "d2min_left": d2min_1,
                    "d2min_right": d2min_2,
                }
            )
            dataset.write_parquet(job.fn(df_path), use_pyarrow=True)
