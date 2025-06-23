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
max_shears = [0.02, 0.05, 0.12]
use_temp = 1.9836e-2

# if we have plotting data cached, grab it now

cache_output = glob.glob(".data/fig6-cache.pkl")
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

        expers = sorted(glob.glob(job.fn("longer_experiments/*/*/traj-fire_period-*.gsd")))

        for exper in expers:
            
            max_shear = utils.extract_between(exper, "max-shear-", "/")
            period = utils.extract_between(exper, "period-", ".gsd")
            temp = utils.extract_between(exper, "temp-", "/")
            sp = Statepoint(max_shear=float(max_shear), period=float(period), temp=float(temp), prep=prep)
            out_path = f"longer_experiments/max-shear-{max_shear}/temp-{temp}/d2min-c_period-{period}.parquet"

            if float(max_shear) not in max_shears:
                continue

            if float(temp) != use_temp:
                continue

            # if len(output[sp]) > 0:
            #     continue
            # take the last row
            lzdf = pl.scan_parquet(job.fn(out_path))
            n_rows = lzdf.select(pl.count()).collect().item()
            # df = pl.read_parquet(job.fn(out_path), columns=["frame", "d2min_irr"], row_count_offset=n_rows-1)
            df = lzdf.select(pl.last(("frame", "d2min_irr", "d2min_rev"))).collect()
            # df = lzdf.select(pl.first(("frame", "d2min_irr"))).collect()
            traj = gsd.hoomd.open(exper, "rb")
            frame = df["frame"][0]
            snap = traj[frame]
            pos = snap.particles.position
            # add pos to df
            # df = df.with_columns({"x", pos[:, 0]})


            output[sp].append({"pos": pos, "df": df})

    with open(".data/fig6-cache.pkl", "wb") as f:
        pickle.dump(output, f)

# some plotting functions
def plot_d2min_field(pos, df, norm=None, cmap=None, ax=None, annotation=None, subtitle=None):
    if ax is None:
        fig, ax = plt.subplots(dpi=100)

    if cmap is None:
        cmap = cm.viridis
    ax.set_aspect("equal")
    ax.scatter(pos[:, 0], pos[:, 1], c=df["d2min_irr"][0], cmap=cmap, s=0.25, norm=norm)
    # remove all axis labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # remove frame boundary
    for spine in ax.spines.values():
        spine.set_visible(False)
    if annotation is not None:
        ax.text(-0.0, 1.0, annotation, transform=ax.transAxes, 
                size=20, weight='bold')
    if subtitle is not None:
        ax.set_title(**subtitle)


def plot_d2min_field_rev(pos, df, norm=None, cmap=None, ax=None, annotation=None, subtitle=None):
    if ax is None:
        fig, ax = plt.subplots(dpi=100)

    if cmap is None:
        cmap = cm.viridis
    ax.set_aspect("equal")
    ax.scatter(pos[:, 0], pos[:, 1], c=df["d2min_rev"][0], cmap=cmap, s=.1, norm=norm)
    # remove all axis labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # remove frame boundary
    for spine in ax.spines.values():
        spine.set_visible(False)
    if annotation is not None:
        ax.text(-0.0, 1.0, annotation, transform=ax.transAxes, 
                size=20, weight='bold')
    if subtitle is not None:
        ax.set_title(**subtitle)

# make a super figure and then spawn subfigures from that
# supsupfig = plt.figure(figsize=(12, 8), dpi=200, layout="constrained")
# supfig = supsupfig.subfigures(1, 2, width_ratios=[3, 1])
supfig = plt.figure(figsize=(9, 8), dpi=200, layout="constrained")
subfigs = supfig.subfigures(2, 1, height_ratios=[1.6666, 1])

fig = subfigs[0]

# TOP FIGURE PANEL
axs = fig.subplots(2, 3)
axs[0, 0].set_ylabel("HTL", fontsize=20)
axs[1, 0].set_ylabel("GQ", fontsize=20)

norm = colors.LogNorm(vmin=1e-2, vmax=1e1)

labels = [["a", "b", "c"], ["d", "e", "f"]]
# cmap = sns.color_palette("mako", as_cmap=True)
cmap = cm.viridis
sm = cm.ScalarMappable(norm, cmap)
for key, val in output.items():
    # print(key)

    # just grab the first entry
    pos = val[0]["pos"]
    df = val[0]["df"]

    # get strain idx
    strain_idx = max_shears.index(key.max_shear)
    prep_idx = 0 if key.prep == "HTL" else 1
    # print(prep_idx, strain_idx)

    if prep_idx == 0:
        subtitle = {"label": f"$\gamma_{{max}} = {key.max_shear*100}\%$", "size": 20}
    else:
        subtitle = None

    plot_d2min_field(pos, df, norm=norm, cmap=cmap, ax=axs[prep_idx, strain_idx], subtitle=subtitle)

# make background white
fig.patch.set_facecolor('white')

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(sm, cax=cbar_ax)
plt.colorbar(sm, ax=axs, label=r"$D^2_{min;strob}$")
# make colorbar label bigger
plt.gcf().axes[-1].yaxis.label.set_size(20)

# BOTTOM FIGURE PANEL
fig = subfigs[1]
axs = fig.subplots(1, 2, sharey=True)
axs = axs.ravel()

# plt.savefig("fig1a.png")
plt.sca(axs[0])
# plt.figure()
bins = np.geomspace(1e-7, 1e2, 25)

cmap = cm.magma

cs = [cmap(0), cmap(0.5), cmap(0.9)]

for key, val in output.items():

    d2min_h, b = np.histogram(val[0]["df"]["d2min_irr"][0], bins=bins)
    d2min_h = d2min_h / np.sum(d2min_h)

    if key.prep == "HTL":
        ls = ":"
    else:
        ls = "-"

    idx = max_shears.index(key.max_shear)

    prep = key.prep

    plt.plot(b[:-1], d2min_h, ls=ls, color=cs[idx], label=f"{prep} {key.max_shear*100}%")

# plt.yscale("log")
plt.xscale("log")
plt.xlabel(r"$D^2_{min;strob}$", fontsize=20)
plt.ylabel("PDF", fontsize=20)



plt.sca(axs[1])
bins = np.geomspace(1e-4, 1e1, 25)

cmap = cm.magma

cs = [cmap(0), cmap(0.5), cmap(0.9)]

for key, val in output.items():

    df = pd.concat

    d2min_h, b = np.histogram(val[0]["df"]["d2min_rev"][0], bins=bins)
    d2min_h = d2min_h / np.sum(d2min_h)

    if key.prep == "HTL":
        ls = ":"
    else:
        ls = "-"

    idx = max_shears.index(key.max_shear)

    prep = key.prep
    if prep == "ESL":
        prep = "GQ"

    plt.plot(b[:-1], d2min_h, ls=ls, color=cs[idx], label=f"{prep}, {key.max_shear*100}%")


x = np.linspace(-1/2*np.pi, (2.5)*np.pi, 100)
y = np.sin(x)
axin = axs[0].inset_axes([0.35, 0.65, 0.3, 0.3])
axin.hlines(0, x[0], x[-1], color="gray", ls="--")
axin.plot(x, y, "k")
axin.plot([0, 2*np.pi], [0, 0], "ro")
axin.set_xticks([])
axin.set_yticks([])
axin.set_xticklabels([])
axin.set_yticklabels([])
for spine in axin.spines.values():
    # spine.set_visible(False)
    # set spine color
    spine.set_color('gray')
axin.set_xlim(x[0], x[-1])
axin.set_ylim(-1.2, 1.2)
axin.set_ylabel(r"$\gamma$", rotation=90)
# set background color
# axin.patch.set_facecolor('gray')

axin = axs[1].inset_axes([0.35, 0.65, 0.3, 0.3])
axin.hlines(0, x[0], x[-1], color="gray", ls="--")
axin.plot(x, y, "k")
axin.plot([.5*np.pi, 1.5*np.pi], [1, -1], "ro")
axin.set_xticks([])
axin.set_yticks([])
axin.set_xticklabels([])
axin.set_yticklabels([])
for spine in axin.spines.values():
    # spine.set_visible(False)
    # set spine color
    spine.set_color('gray')
axin.set_xlim(x[0], x[-1])
axin.set_ylim(-1.2, 1.2)
axin.set_ylabel(r"$\gamma$", rotation=90)

# plt.yscale("log")
plt.xscale("log")
plt.xlabel(r"$D^2_{min;ptp}$", fontsize=20)
# plt.ylabel("PDF", fontsize=20)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=r"(PREP, $\gamma_{max})$")
# plt.tight_layout()

plt.savefig("FIG6.png")