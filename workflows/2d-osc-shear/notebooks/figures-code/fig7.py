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
# import hoomd
import schmeud
import schmeud._schmeud as schmeud_rs

from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

import polars as pl

from monk import utils, workflow

parent = pathlib.Path(os.getcwd()).parent.parent / "config.yaml"
config = workflow.get_config(parent.as_posix())
parent, config

project: signac.Project = signac.get_project(root=config['root'])
project.doc

@dataclass(frozen=True, eq=True)
class Statepoint:
    max_shear: float
    period: float
    temp: float
    prep: str

ts = [0.0019836, 0.019836, 0.049591, 0.19836]
t = ts[2]
s = 0.05

cache_output = glob.glob(".data/fig7.0-cache.pkl")
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

        expers = sorted(glob.glob(job.fn("longer_experiments/*/*/traj-fire_period-*.gsd")))

        for exper in tqdm(expers):
            max_shear = utils.extract_between(exper, "max-shear-", "/")
            period = utils.extract_between(exper, "period-", ".gsd")
            temp = utils.extract_between(exper, "temp-", "/")
            sp = Statepoint(max_shear=float(max_shear), period=float(period), temp=float(temp), prep=prep)
            out_path = job.fn(f"longer_experiments/max-shear-{max_shear}/temp-{temp}/msd-strob_period-{period}.parquet")

            if sp.max_shear != s or sp.temp not in [9.9182e-02, 1.9836e-02]: # 0.19836:
                continue

            if job.isfile(out_path):
                dataset = pl.read_parquet(job.fn(out_path), columns=["frame_i", "frame_j", "ave_msd"])
                # dataset.drop_in_place("msd")
                # dataset.drop_in_place("id")
                output0[sp].append(dataset)

    with open(".data/fig7.0-cache.pkl", "wb") as f:
        pickle.dump(output0, f)

cache_output = glob.glob(".data/fig7.1-cache.pkl")
if len(cache_output) > 0:
    with open(cache_output[0], "rb") as f:
        output1 = pickle.load(f)
else:
    output1 = defaultdict(list)

    for job in project:
        print(job)
        prep = job.sp["prep"]
        # if prep != "HTL":
        #     continue

        expers = sorted(glob.glob(job.fn("longest_experiments/*/*")))

        for exper in tqdm(expers):
            max_shear = utils.extract_between(exper, "max-shear-", "/")
            period = 1000
            # iter = utils.extract_between(exper, "iter-", ".gsd")
            temp = utils.extract_between(exper, "temp-", "/")
            sp = Statepoint(max_shear=float(max_shear), period=float(period), temp=float(temp), prep=prep)
            out_path = f"longest_experiments/max-shear-{max_shear}/temp-{temp}/msd-strob-total_period-{period}.parquet"

            if job.isfile(out_path):
                dataset = pl.read_parquet(job.fn(out_path), columns=["frame_i", "frame_j", "ave_msd"])
                # dataset.drop_in_place("msd")
                # dataset.drop_in_place("id")
                output1[sp].append(dataset)

    with open(".data/fig7.1-cache.pkl", "wb") as f:
        pickle.dump(output1, f)

cache_output = glob.glob(".data/fig7.2-cache.pkl")
if len(cache_output) > 0:
    with open(cache_output[0], "rb") as f:
        output2 = pickle.load(f)
else:
    output2 = defaultdict(list)

    # dataset = None

    for job in project:
        print(job)
        prep = job.sp["prep"]
        # print(prep)

        # if prep != "HTL":
        #     continue

        expers = sorted(glob.glob(job.fn("longer_experiments/*/*/traj-fire_period-*.gsd")))
        # print(expers)
        for exper in expers:
            max_shear = utils.extract_between(exper, "max-shear-", "/")
            period = utils.extract_between(exper, "period-", ".gsd")
            temp = utils.extract_between(exper, "temp-", "/")
            sp = Statepoint(max_shear=float(max_shear), period=float(period), temp=float(temp), prep=prep)
            out_path = job.fn(f"longer_experiments/max-shear-{max_shear}/temp-{temp}/msd-strob_period-{period}.parquet")
            # print(out_path)
            if sp.max_shear != s or sp.temp != t: # 0.19836:
                continue

            if sp in output2:
                continue
            
            # print(out_path)
            files = sorted(glob.glob(out_path))
            traj = gsd.hoomd.open(exper, "rb")
            
            data = []
            # dataset = []
            print(files)
            for file in files:
                # print("good")
                dataset = pl.read_parquet(file, columns=["frame_i", "frame_j", "msd"], use_pyarrow=True)
                dataset = dataset.filter(pl.col("frame_i") == 2039)
                data.append(dataset)
                
                # dataset = dataset.filter(pl.col("frame_i") == 2039)
                # dataset.drop_in_place("msd")
                # dataset.drop_in_place("id")
            data = {"df": data, "traj": traj}
            output2[sp].append(data)

    with open(".data/fig7.2-cache.pkl", "wb") as f:
        pickle.dump(output2, f)


cache_output = glob.glob(".data/fig7.3-cache.pkl")
if len(cache_output) > 0:
    with open(cache_output[0], "rb") as f:
        output3 = pickle.load(f)
else:
    output3 = defaultdict(list)

    for job in project:
        print(job)
        prep = job.sp["prep"]

        expers = sorted(glob.glob(job.fn("longer_experiments/*/*/traj-fire_period-*.gsd")))

        for exper in tqdm(expers):
            max_shear = utils.extract_between(exper, "max-shear-", "/")
            period = utils.extract_between(exper, "period-", ".gsd")
            temp = utils.extract_between(exper, "temp-", "/")
            if float(temp) not in [9.9182e-02, 1.9836e-02]:
                continue
            sp = Statepoint(max_shear=float(max_shear), period=float(period), temp=float(temp), prep=prep)
            out_path = f"longer_experiments/max-shear-{max_shear}/temp-{temp}/msd-strob_period-{period}.parquet"

            if job.isfile(out_path):
                dataset = pl.read_parquet(job.fn(out_path), columns=["frame_i", "frame_j", "ave_msd"])
                # dataset.drop_in_place("msd")
                # dataset.drop_in_place("id")
                output3[sp].append(dataset)


    with open(".data/fig7.3-cache.pkl", "wb") as f:
        pickle.dump(output3, f)

from scipy.ndimage import uniform_filter1d
from matplotlib import colors

supsupfig = plt.figure(figsize=(9, 6), dpi=200, layout="constrained")
fig = supsupfig.subfigures(1, 2, width_ratios=[1, 2])
# fig = supfig.subfigures(1, 2, width_ratios=[1, 2])
axs = fig[0].subplots(2, 1)

ax = axs[0]
plt.sca(ax)

# fig = plt.figure(figsize=(8, 8), dpi=100)
# plt.gca().set_aspect('equal')

temps = set()
for key, data in output0.items():
    # print(key.temp)
    temps.add(key.temp)
temps

temps_loc = sorted(list(temps))

cmap = cm.viridis
cs = [cmap(0), cmap(0.25), cmap(0.5), cmap(0.75)]

for key, data in output0.items():
    pair = key.prep
    # print(key)
    if key.temp != temps_loc[0] or key.max_shear != 0.05:
        continue
    # print(key.temp)
    if pair == "ESL":
        ls = "-"
    else:
        ls = ":"
    tws = [[], [], [], []]
    # colors = ["r", "b", "g", "k"]
    waits = [0, 50, 100, 150]
    for d in data:
        for idx, (_, di) in enumerate(d.group_by("frame_i", maintain_order=True)):
            # print(idx)
            tws[idx].append(di[:]["ave_msd"].to_numpy())
            # break
    tws = [np.mean(tw, axis=0) for tw in tws]
    for tw, c, w in zip(tws, cs, waits):
        # get log-log slope
        tw = uniform_filter1d(tw, 10)
        x = np.log(np.arange(len(tw)))[20:]
        y = np.log(tw)[20:]
        slope = np.polyfit(x, y, 1)[0]
        # print(slope)
        # plt.plot(x, y)
        plt.plot(tw, label=f"{pair} {w:.2e} {slope:.2}", ls=ls, color=c)
plt.ylabel("MSD", size="x-large")
plt.xlabel("Cycle", size="x-large")
# plt.yscale('log')
# plt.xscale('log')
# plt.legend(title=r"prep, $t_w$, $D$", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.ylim(0, 0.05)
# plt.savefig("msd-0.05-0.25-log.png", dpi=300, bbox_inches='tight')
# fig = plt.figure(figsize=(8, 8), dpi=100)
# plt.gca().set_aspect('equal')

axin = ax.inset_axes([0.05, 0.65, 0.35, 0.35])

temps = set()
for key, data in output1.items():
    # print(key.temp)
    temps.add(key.temp)
temps

temps_loc = sorted(list(temps))
print(temps_loc)

cmap = cm.viridis
# colors = [cmap(0), cmap(0.25), cmap(0.5), cmap(0.75)]
norm = colors.Normalize(vmin=0, vmax=10)

for key, data in output1.items():
    prep = key.prep
    # print(key)
    if key.temp != temps_loc[0] or key.max_shear != 0.05:
        continue
    # print(key.temp)
    if prep == "ESL":
        ls = "-"
    else:
        ls = ":"
    # tws = [[], [], [], []]
    # colors = ["r", "b", "g", "k"]
    waits = [0, 50, 100, 150]
    # for d in data:
    ds = []
    m = 2000
    for d in data:
        if len(d) == 0:
            continue
        ds.append(d["ave_msd"].to_numpy())
        
        m = min(m, len(d))
                # break
        # tws = [np.mean(tw, axis=0) for tw in tws]
        # for tw, c, w in zip(tws, colors, waits):
        #     # get log-log slope
    ds = [d[:m] for d in ds]
    tws = np.mean(ds, axis=0)
    tw = uniform_filter1d(tws, 50)
    # tw = tws
    x = np.log(np.arange(len(tw)))[1000:2000]
    y = np.log(tw)[1000:2000]
    slope = np.polyfit(x, y, 1)[0]
    # print(prep, slope)
    #     # plt.plot(x, y)
    c = cmap(norm(0))
    x  = np.arange(len(tw))
    axin.plot(x[50:], tw[50:], label=f"{prep} {slope:.3}", ls=ls, color=c)
# plt.sca(axin)
# plt.ylabel("MSD", size="x-large")
# plt.xlabel("Cycle", size="x-large")
axin.set_yscale('log')
axin.set_xscale('log')
axin.set_xlim(10, 2000)
axin.set_ylim(5e-2, 2e1)
# axin.set_xticks([])
# axin.set_yticks([])
# axin.set_xticklabels([])
# axin.set_yticklabels([])
X_TICK_PADDING = 0
Y_TICK_PADDING = -5

xticks = [*axin.xaxis.get_major_ticks(), *axin.xaxis.get_minor_ticks()]
yticks = [*axin.yaxis.get_major_ticks(), *axin.yaxis.get_minor_ticks()]

for tick in xticks:
    tick.set_pad(X_TICK_PADDING)

for tick in yticks:
    tick.set_pad(Y_TICK_PADDING)
    # justify left
    tick.label1.set_horizontalalignment('left')

# set tick fontsize small
for tick in xticks:
    tick.label1.set_fontsize(6)
for tick in yticks:
    tick.label1.set_fontsize(6)

axin.plot([250, 1500], np.array([.5, 3]), "k--")

temps = set()
for key, data in output0.items():
    # print(key.temp)
    temps.add(key.temp)
temps

temps_loc = sorted(list(temps))

cmap = cm.viridis
cs = [cmap(0), cmap(0.25), cmap(0.5), cmap(0.75)]

ax = axs[1]
plt.sca(ax)

for key, data in output0.items():
    pair = key.prep
    # print(key)
    if key.temp != temps_loc[1] or key.max_shear != 0.05:
        continue
    # print(key.temp)
    if pair == "ESL":
        ls = "-"
    else:
        ls = ":"
    tws = [[], [], [], []]
    # colors = ["r", "b", "g", "k"]
    waits = [0, 50, 100, 150]
    for d in data:
        for idx, (_, di) in enumerate(d.group_by("frame_i", maintain_order=True)):
            # print(idx)
            tws[idx].append(di[:]["ave_msd"].to_numpy())
            # break
    tws = [np.mean(tw, axis=0) for tw in tws]
    for tw, c, w in zip(tws, cs, waits):
        # get log-log slope
        tw = uniform_filter1d(tw, 10)
        x = np.log(np.arange(len(tw)))[20:]
        y = np.log(tw)[20:]
        slope = np.polyfit(x, y, 1)[0]
        # print(slope)
        # plt.plot(x, y)
        plt.plot(tw, label=f"{pair} {w:.2e} {slope:.2}", ls=ls, color=c)
plt.ylabel("MSD", size="x-large")
plt.xlabel("Cycle", size="x-large")

# plt.plot()
# plt.legend(title=r"prep $\alpha$")

rightsupfig = fig[1]
subfigs = rightsupfig.subfigures(2, 1)

axs = subfigs[0].subplots(1, 2)
# set aspect 1
for ax in axs:
    ax.set_aspect('equal', adjustable='box')
    for spine in ax.spines.values():
        spine.set_visible(False)

for key, data in output2.items():
    # print(key)

    dataset = data[0]["df"]
    traj = data[0]["traj"]
    # print(dataset[0])
    row = dataset[0][-1]

    frame = row["frame_j"][0]
    frame_i = row["frame_i"][0]
    msd = row["msd"][0]

    # print(traj, frame_i, frame)

    snap = traj[int(frame)]

    pos = snap.particles.position

    orig_snap = traj[int(frame_i)]
    orig_pos = orig_snap.particles.position


    # disp = np.sqrt(msd).to_numpy()

    box = freud.Box.from_box(orig_snap.configuration.box)

    disp = box.wrap(pos - orig_pos)
    mdisp = np.mean(disp, axis=0)
 

    # query = freud.locality.AABBQuery.from_system(orig_snap)
    # nlist = query.query(orig_pos, dict(r_max=2.0)).toNeighborList()

    disp = np.linalg.norm(disp - mdisp, axis=1)

    if key.prep == "HTL":
        pos_htl = orig_pos
        disp_htl = disp
    else:
        pos_esl = orig_pos
        disp_esl = disp

norm = colors.LogNorm(vmin=5e-3)

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
        

plt.sca(axs[0])
plt.scatter(pos_htl[:, 0], pos_htl[:, 1], c=disp_htl, cmap="jet", norm=norm, s=.25)
plt.title("HTL", size="xx-large")

plt.sca(axs[1])
plt.scatter(pos_esl[:, 0], pos_esl[:, 1], c=disp_esl, cmap="jet", norm=norm, s=.25)
plt.title("GQ", size="xx-large")

# make colorbar for the entire figure
cbar = fig[1].colorbar(plt.cm.ScalarMappable(norm=norm, cmap="jet"), ax=axs, orientation="vertical",fraction=0.02, pad=0.01)
cbar.set_label(r"$\Delta r$", size="xx-large")

axs = subfigs[1].subplots(1, 2, sharey=True)
axs[0].set_ylabel(r"$G(\Delta r)$", size="xx-large")

N = traj[0].particles.N

hnorm = colors.Normalize(vmin=0, vmax=140)
cmap = cm.jet

for key, data in output2.items():
    dataset = data[0]["df"][0]
    traj = data[0]["traj"]

    if key.prep == "HTL":
        plt.sca(axs[0])
    else:
        plt.sca(axs[1])

    for idx in range(len(dataset))[19::20]:

        row = dataset[idx]

        frame = row["frame_j"][0]
        frame_i = row["frame_i"][0]
        print(frame_i, frame)
        # print(frame, frame_i)
        msd = row["msd"][0]

        snap = traj[int(frame)]

        pos = snap.particles.position

        orig_snap = traj[int(frame_i)]
        orig_pos = orig_snap.particles.position

        # query = freud.locality.AABBQuery.from_system(orig_snap)
        # nlist = query.query(orig_pos, dict(r_max=2.0)).toNeighborList()

        # box = freud.Box.from_box(snap.configuration.box)

        # disp = box.wrap(pos - orig_pos)
        # mdisp = np.zeros_like(disp)

        # for jdx, (seg, nn) in enumerate(zip(nlist.segments, nlist.neighbor_counts)):
        #     idxs = nlist.point_indices[seg:seg+nn]
        #     mean_disp = np.mean(disp[idxs], axis=0)
        #     mdisp[jdx] = mean_disp

        # disp = np.linalg.norm(disp - mdisp, axis=1)

        disp = np.sqrt(msd).to_numpy()
        
        plt.hist(disp, bins=np.linspace(0, 5, 50), histtype="step", color=cmap(hnorm(idx)), density=True, label=f"{idx+1}")

for ax in axs:
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta r$", size="xx-large")

plt.sca(axs[1])
plt.legend(title="CYCLE")

# plt.ylim(0, 0.05)
plt.savefig("FIG7.png", dpi=300, bbox_inches='tight')