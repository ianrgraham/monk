# bring in all imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm, colors

import glob
import gc
import warnings
import copy
import pathlib
import pickle

import os

from scipy import stats
import signac
import freud
import gsd.hoomd
import schmeud
import schmeud._schmeud as schmeud_rs

from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

import polars as pl
import seaborn as sns

from monk import utils, workflow

# grab project

parent = pathlib.Path(os.getcwd()).parent.parent / "config.yaml"
config = workflow.get_config(parent.as_posix())

project: signac.Project = signac.get_project(root=config['root'])

@dataclass(frozen=True, eq=True)
class Statepoint:
    max_shear: float
    period: float
    temp: float
    prep: str

# only look at lowest temperature data
# max_shears = [0.04, 0.05, 0.06]
use_temp = 0
# use_temp = 1.9836e-3
use_temps = [1.9836e-3, 1.9836e-2, 9.9182e-02]

# if we have plotting data cached, grab it now

cache_output = glob.glob(".data/fig2.1-cache.pkl")
if len(cache_output) > 0:
    with open(cache_output[0], "rb") as f:
        output0 = pickle.load(f)
else:
    output0 = defaultdict(list)

    for job in project:
        print(job)
        prep = job.sp["prep"]
        # if prep != "HTL":
        #     continue

        expers = sorted(glob.glob(job.fn("aqs_experiments_longer/*/*/traj_period-*.gsd")))
        print(len(expers))

        for exper in expers:
            
            max_shear = utils.extract_between(exper, "max-shear-", "/")
            period = utils.extract_between(exper, "period-", "_")
            temp = utils.extract_between(exper, "temp-", "/")
            sp = Statepoint(max_shear=float(max_shear), period=float(period), temp=float(temp), prep=prep)
            out_path = f"aqs_experiments_longer/max-shear-{max_shear}/temp-{temp}/d2min-c_period*.parquet"
            out_path = glob.glob(job.fn(out_path))
            if len(out_path) == 0:
                continue
            out_path = out_path[0]

            # if float(max_shear) not in max_shears:
            #     continue

            if float(temp) != use_temp:
                continue

            if not os.path.exists(out_path):
                print("skipping", job)
                continue

            # if len(output[sp]) > 0:
            #     continue
            # take the last row
            lzdf = pl.scan_parquet(out_path)
            n_rows = lzdf.select(pl.count()).collect().item()
            print(n_rows)
            # df = pl.read_parquet(job.fn(out_path), columns=["frame", "d2min_irr"], row_count_offset=n_rows-1)
            # df = lzdf.select(pl.last(("frame", "d2min_irr",
            # "d2min_rev"))).collect()
            if n_rows < 99:
                continue
            df = lzdf.slice(95,1).collect()
            # df = lzdf.select(pl.first(("frame", "d2min_irr"))).collect()
            traj = gsd.hoomd.open(exper, "rb")
            frame = df["frame"][0]
            snap = traj[frame]
            pos = snap.particles.position
            # add pos to df
            # df = df.with_columns({"x", pos[:, 0]})

            # print("woo!", job, sp)
            output0[sp].append({"pos": pos, "df": df})
    with open(".data/fig2.1-cache.pkl", "wb") as f:
        pickle.dump(output0, f)

cache_output = glob.glob(".data/fig2.2-cache.pkl")
if len(cache_output) > 0:
    with open(cache_output[0], "rb") as f:
        output = pickle.load(f)
else:
    output = defaultdict(list)

    for job in project:
        print(job)
        prep = job.sp["prep"]
        # if prep != "HTL":
        #     continue

        expers = sorted(glob.glob(job.fn("longer_experiments/*/*/traj_period-*.gsd")))
        print(len(expers))

        for exper in expers:
            
            max_shear = utils.extract_between(exper, "max-shear-", "/")
            period = utils.extract_between(exper, "period-", ".gsd")
            temp = utils.extract_between(exper, "temp-", "/")
            sp = Statepoint(max_shear=float(max_shear), period=float(period), temp=float(temp), prep=prep)
            out_path = f"longer_experiments/max-shear-{max_shear}/temp-{temp}/d2min-c_period*.parquet"
            out_path = glob.glob(job.fn(out_path))
            if len(out_path) == 0:
                continue
            out_path = out_path[0]

            # if float(max_shear) not in max_shears:
            #     continue

            if float(temp) not in use_temps:
                continue

            if not os.path.exists(out_path):
                print("skipping", job)
                continue

            # if len(output[sp]) > 0:
            #     continue
            # take the last row
            lzdf = pl.scan_parquet(out_path)
            n_rows = lzdf.select(pl.count()).collect().item()
            print(n_rows)
            # df = pl.read_parquet(job.fn(out_path), columns=["frame", "d2min_irr"], row_count_offset=n_rows-1)
            # df = lzdf.select(pl.last(("frame", "d2min_irr",
            # "d2min_rev"))).collect()
            if n_rows < 99:
                continue
            df = lzdf.slice(198,1).collect()
            # df = lzdf.select(pl.first(("frame", "d2min_irr"))).collect()
            traj = gsd.hoomd.open(exper, "rb")
            frame = df["frame"][0]
            snap = traj[frame]
            pos = snap.particles.position
            # add pos to df
            # df = df.with_columns({"x", pos[:, 0]})

            # print("woo!", job, sp)
            output[sp].append({"pos": pos, "df": df})
    with open(".data/fig2.2-cache.pkl", "wb") as f:
        pickle.dump(output, f)

fig, axs = plt.subplots(1, 3, figsize=(9, 3), dpi=200, sharey=True)

plt.sca(axs[0])

def plot_for_temp(output, ax, temp):
    plt.sca(ax)
    xs_gq = []
    ys_gq = []
    xs_htl = []
    ys_htl = []

    for key, val in output.items():

        if key.temp != temp:
            continue

        all_vals_rev = np.vstack([v["df"]["d2min_rev"][0] for v in val])
        all_vals_irr = np.vstack([v["df"]["d2min_irr"][0] for v in val])

        # irr_over_rev = np.mean(np.log(all_vals_irr)) -
        # np.mean(np.log(all_vals_rev))
        irr_over_rev = np.mean(all_vals_irr) / np.mean(all_vals_rev)

        if key.prep == "ESL":
            xs_gq.append(key.max_shear)
            ys_gq.append(irr_over_rev)
        else:
            xs_htl.append(key.max_shear)
            ys_htl.append(irr_over_rev)

    xs_gq = np.array(xs_gq)
    ys_gq = np.array(ys_gq)
    xs_htl = np.array(xs_htl)
    ys_htl = np.array(ys_htl)

    # print(xs_gq, ys_gq)
    # print(xs_htl, ys_htl)

    # sort arrays by xs
    sort_idxs_gq = np.argsort(xs_gq)
    sort_idxs_htl = np.argsort(xs_htl)

    xs_gq = xs_gq[sort_idxs_gq]
    ys_gq = ys_gq[sort_idxs_gq]
    xs_htl = xs_htl[sort_idxs_htl]
    ys_htl = ys_htl[sort_idxs_htl]

    ax.plot(xs_gq, ys_gq, "o-", label="GQ")
    ax.plot(xs_htl, ys_htl, "o:", label="HTL")

plot_for_temp(output0, axs[0], use_temp)
plot_for_temp(output, axs[1], use_temps[0])
plot_for_temp(output, axs[2], use_temps[1])
# plot_for_temp(output, axs[3], use_temps[2])

# axs[0].set_ylabel()
# plt.yscale('log')

plt.savefig("FIG2.png")