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

# prefix = "aqs_therm_"
prefix = "longer_"

@dataclass(frozen=True, eq=True)
class Statepoint:
    max_shear: float
    period: float
    temp: float
    prep: str

# only look at lowest temperature data
max_shears = [0.02, 0.05, 0.12]
use_temp = 1.9836e-3
# use_temp = 4.9591e-02

# if we have plotting data cached, grab it now

cache_output = glob.glob(".data/fig5-cache.pkl")
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

        expers = sorted(glob.glob(job.fn(f"{prefix}experiments/*/*/traj-fire_period-*.gsd")))

        for exper in expers:
            
            max_shear = utils.extract_between(exper, "max-shear-", "/")
            period = utils.extract_between(exper, "period-", ".gsd")
            temp = utils.extract_between(exper, "temp-", "/")
            sp = Statepoint(max_shear=float(max_shear), period=float(period), temp=float(temp), prep=prep)
            out_path = f"{prefix}experiments/max-shear-{max_shear}/temp-{temp}/d2min-c_period-{period}.parquet"
            soft_path = f"{prefix}experiments/max-shear-{max_shear}/temp-{temp}/strob-softness_period-{period}.parquet"

            # if float(max_shear) not in max_shears:
            #     continue

            if float(temp) != use_temp:
                continue

            # if len(output[sp]) > 0:
            #     continue
            # take the last row
            # lzdf = pl.scan_parquet(job.fn(out_path))
            # n_rows = lzdf.select(pl.count()).collect().item()
            df = pl.read_parquet(job.fn(out_path), columns=["frame", "d2min_irr"])
            df_soft = pl.read_parquet(job.fn(soft_path))
            # df = lzdf.select(pl.col("d2min_irr")).collect()
            # df = lzdf.select(pl.first(("frame", "d2min_irr"))).collect()
            # traj = gsd.hoomd.open(exper, "rb")
            # frame = df["frame"][0]
            # snap = traj[frame]
            # pos = snap.particles.position
            # add pos to df
            # df = df.with_columns({"x", pos[:, 0]})


            output[sp].append({"df": df, "soft_df": df_soft})

    with open(".data/fig5-cache.pkl", "wb") as f:
        pickle.dump(output, f)



# make a super figure and then spawn subfigures from that
# supsupfig = plt.figure(figsize=(12, 8), dpi=200, layout="constrained")
# supfig = supsupfig.subfigures(1, 2, width_ratios=[3, 1])
# supfig = plt.figure(figsize=(9, 8), dpi=200, layout="constrained")
# subfigs = supfig.subfigures(1, 1, height_ratios=[1, 1])

# fig = subfigs[0]
fig = plt.figure(figsize=(9, 7.5), dpi=200, layout="constrained")
subfigs = fig.subfigures(2, 1, height_ratios=[3.5, 4])


# TOP FIGURE PANEL
fig1 = subfigs[0]
axs = fig1.subplots(1, 3, sharey=False)
fig2 = subfigs[1]
axs2 = fig2.subplots(1, 2)
cp = sns.color_palette("coolwarm_r",6)
# axs[0, 0].set_ylabel("HTL", fontsize=20)
# axs[1, 0].set_ylabel("GQ", fontsize=20)

norm = colors.LogNorm(vmin=1e-2, vmax=1e1)

labels = [["a", "b", "c"], ["d", "e", "f"]]
# cmap = sns.color_palette("mako", as_cmap=True)
cmap = cm.viridis
sm = cm.ScalarMappable(norm, cmap)
annotations = ["a", "b", "c"]
for key, val in output.items():
    # print(key)

    # just grab the first entry
    # pos = val[0]["pos"]
    
    if key.max_shear not in max_shears:
        # print(key.max_shear)
        continue
    # print('cool!')
    strain_idx = max_shears.index(key.max_shear)
    # prep_idx = 0 if key.prep == "HTL" else 1
    # # print(prep_idx, strain_idx)

    # if prep_idx == 0:
    #     subtitle = {"label": f"$\gamma_{{max}} = {key.max_shear*100}\%$", "size": 20}
    # else:
    #     subtitle = None
    md2m = []
    for v in val:
        df = v["df"]
        d2min = np.vstack(df["d2min_irr"].to_numpy())[:80]

        mean_d2min = np.mean(d2min, axis=1)
        md2m.append(mean_d2min)

    md2m = np.vstack(md2m)
    # print(md2m.shape)
    mean_d2min = np.mean(md2m, axis=0)

    if key.prep == "HTL":
        ls = "--"
        m = "."
        c = cp[0]
    else:
        ls = "-"
        m = "."
        c = cp[-1]

    axs[strain_idx].plot(mean_d2min, ls=ls, marker=m, color=c)
    axs[strain_idx].set_title(fr"$\gamma_{{max}} = {key.max_shear*100}\%$", size=20)
    ax = axs[strain_idx]
    ax.text(-0.05, 1.05, annotations[strain_idx], transform=ax.transAxes, weight="bold", size=15)
    # axs[strain_idx].set_yscale('log')
    # plot_d2min_field(pos, df, norm=norm, cmap=cmap, ax=axs[prep_idx,
    # strain_idx], subtitle=subtitle)
    
gammas = []
out = []

gamma_gq = []
out_gq = []

softs = []
softs_gq = []
for key, val in output.items():
    # print(key)

    # just grab the first entry
    # pos = val[0]["pos"]
    

    # get strain idx
    # strain_idx = max_shears.index(key.max_shear)
    # prep_idx = 0 if key.prep == "HTL" else 1
    # # print(prep_idx, strain_idx)

    # if prep_idx == 0:
    #     subtitle = {"label": f"$\gamma_{{max}} = {key.max_shear*100}\%$", "size": 20}
    # else:
    #     subtitle = None
    md2m = []
    soft = []
    for v in val:
        df = v["df"]
        soft_df = v["soft_df"]
        md2m.append(np.mean(df["d2min_irr"].to_numpy()[80]))
        soft.append(np.mean(soft_df["soft"].to_numpy()[80]))

    mean_d2min = np.mean(md2m)
    mean_soft = np.mean(soft)
    if key.prep == "HTL":
        out.append(mean_d2min)
        gammas.append(key.max_shear)
        softs.append(mean_soft)
    else:
        out_gq.append(mean_d2min)
        gamma_gq.append(key.max_shear)
        softs_gq.append(mean_soft)

gammas, out, softs = zip(*sorted(zip(gammas, out, softs)))
axs2[0].plot(gammas, out, ls="--", marker=".", color=cp[0], label="HTL")
axs2[1].plot(gammas, softs, ls="--", marker=".", color=cp[0])


gamma_gq, out_gq, softs_gq = zip(*sorted(zip(gamma_gq, out_gq, softs_gq)))
axs2[0].plot(gamma_gq, out_gq, ls="-", marker=".", color=cp[-1], label="GQ")
axs2[1].plot(gamma_gq, softs_gq, ls="-", marker=".", color=cp[-1])

legend = axs2[0].legend(title="PREP", fontsize="x-large")
# set title fontsize
plt.setp(legend.get_title(),fontsize='x-large')

axs[0].set_ylabel(r"$\langle D^2_{min;strob} \rangle$", fontsize=20)
for ax in axs:
    ax.set_xlabel("Cycle", fontsize=20)
axs2[0].set_ylabel(r"$\langle D^2_{min;strob} \rangle$", fontsize=20)
axs2[0].set_xlabel(r"$\gamma_{max}$", fontsize=20)
axs2[1].set_ylabel(r"$\langle S \rangle$", fontsize=20)
axs2[1].set_xlabel(r"$\gamma_{max}$", fontsize=20)

axs2[0].text(-0.05, 1.05, "d", transform=axs2[0].transAxes, weight="bold", size=15)
axs2[1].text(-0.05, 1.05, "e", transform=axs2[1].transAxes, weight="bold", size=15)
# axs[3].set_yscale("log")

plt.savefig("FIG5.png")