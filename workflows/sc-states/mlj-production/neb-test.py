# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib import colors, cm
import glob
from datetime import datetime, timedelta

from numba import njit, vectorize, float32

from typing import Callable, Optional, Union, List

import hoomd
from hoomd.neb_plugin import neb
import gsd.hoomd
import freud
import schmeud
from schmeud._schmeud import dynamics as schmeud_dynamics

import sys
import time
import pickle
import gc
import warnings
import copy
import pathlib
from collections import defaultdict

import os
import sys

import signac

from dataclasses import dataclass
from dataclasses_json import dataclass_json

import fresnel
import PIL.Image
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["axes.labelsize"] = "xx-large"
from monk import nb, prep, pair, render, utils, workflow, grid

config = workflow.get_config()
project: signac.Project = signac.get_project(root=config['root'])
project.doc

# %%
import hoomd.pair_plugin.pair as pair_plugin

# %%
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["axes.labelsize"] = "xx-large"

# %%
from monk import nb, prep, pair, render, utils, workflow, grid

config = workflow.get_config()

# %%
project: signac.Project = signac.get_project(root=config['root'])
project.doc

# %%
for idx, job in enumerate(project.find_jobs({"delta": 0.0, "replica": 0})):
    print(idx, job.doc)
    doc = job.doc
    sp = job.sp

    dt = project.doc["dt"]

    runs = sorted(glob.glob(job.fn("short_runs/temp-*/")))[2:]
    for run in runs:
        print(run)
        temp = float(utils.extract_between(run, "temp-", "/"))
        fire_file = run + "fire_traj.gsd"
        # print(run)
        traj = gsd.hoomd.open(fire_file, "rb")
        df = pd.read_parquet(run + "struct-descr.parquet")

        break

# %%
idx = 0
tdf = df[df.frame == idx]
len(tdf[tdf.phop > 0.2])

# %%
df.head()

# %%
sdf = df[df.frame == 0]
smin = sdf.phop.min()
smax = sdf.phop.max()
sdf.hist("phop", bins=np.logspace(np.log10(smin), np.log10(smax), 20))
plt.xscale('log')

# %%
nlist = hoomd.md.nlist.Tree(0.3)
mlj = pair.KA_ModLJ(nlist, 0.0)
forces = [mlj]
dev = "gpu"
filter = hoomd.filter.All()

snap2snap = hoomd.Snapshot.from_gsd_snapshot
device = hoomd.device.GPU()

neb_driver = neb.NEBDriver(snap2snap(traj[10], device.communicator), snap2snap(traj[11], device.communicator), n_images=4, forces=forces, filter=filter, device=dev)
neb_driver.k = 100.0

# %%
# for sim in neb_driver.nodes:
#     sim.device._cpp_exec_conf.setCUDAErrorChecking(True)

# %%
start = neb_driver._neb_sims[0]
end = neb_driver._neb_sims[-1]

# %%
energy = []

end.operations.integrator.nudge = False
for i in range(1):
    start_time = time.time()
    end.run(50)
    print(f"{i}:", time.time() - start_time)
    energy.append(end.operations.integrator.energy)

plt.plot(energy)

# %%

start.operations.integrator.nudge = False
energy = []
for i in range(1):
    start_time = time.time()
    start.run(50)
    print(f"{i}:", time.time() - start_time)
    energy.append(start.operations.integrator.energy)
# start.operations.integrator.nudge = True


plt.plot(energy)
# plt.yscale('log')

# %%
start.operations.integrator.nudge = True
end.operations.integrator.nudge = True

# %%
def d_omega(snap1, snap2):
    box = freud.box.Box.from_box(snap1.configuration.box)
    pos1 = snap1.particles.position
    pos2 = snap2.particles.position
    diff = box.wrap(pos2 - pos1)
    return np.linalg.norm(diff)

# %%
# for sim in neb_driver._neb_sims:
#     start.operations.integrator.nudge = False

# %%
cmap = cm.viridis
runs = 1
norm = colors.Normalize(vmin=0, vmax=runs)

fig, axs = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)

energy = []
dists = []
other_dists = []
nodes = neb_driver.nodes
for i in range(len(nodes)-1):
    # print(len(sim.operations.integrator.forces))
    # sim.operations.integrator.reset()
    ps_dist = d_omega(nodes[i].state.get_snapshot(), nodes[i+1].state.get_snapshot())
    dists.append(ps_dist)
    energy.append(nodes[i].operations.integrator.forces[0].energy)
energy.append(nodes[-1].operations.integrator.forces[0].energy)

iter = list(range(1, len(nodes)))
for i in iter:
    ps_dist = d_omega(nodes[0].state.get_snapshot(), nodes[i].state.get_snapshot())
    other_dists.append(ps_dist)

axs[0].plot(energy, color=cmap(norm(0)))
axs[1].plot(dists, color=cmap(norm(0)))
axs[2].plot(iter, other_dists, color=cmap(norm(0)))

for j in range(1, runs+1):
    start = time.time()
    neb_driver.run(50)
    print(f"{j}:", time.time() - start)
    energy = []
    dists = []
    other_dists = []
    nodes = neb_driver.nodes
    for i in range(len(nodes)-1):
        # print(len(sim.operations.integrator.forces))
        # sim.operations.integrator.reset()
        ps_dist = d_omega(nodes[i].state.get_snapshot(), nodes[i+1].state.get_snapshot())
        dists.append(ps_dist)
        energy.append(nodes[i].operations.integrator.forces[0].energy)
    energy.append(nodes[-1].operations.integrator.forces[0].energy)

    iter = list(range(1, len(nodes)))
    for i in iter:
        ps_dist = d_omega(nodes[0].state.get_snapshot(), nodes[i].state.get_snapshot())
        other_dists.append(ps_dist)

    axs[0].plot(energy, color=cmap(norm(j)))
    axs[1].plot(dists, color=cmap(norm(j)))
    axs[2].plot(iter, other_dists, color=cmap(norm(j)))
axs[0].set_ylabel("Energy")
axs[0].set_xlabel("Node")
axs[1].set_ylabel(r"$|d\Omega|$")
axs[1].set_xlabel("Segment")
axs[2].set_ylabel(r"$|d\Omega_{0,i}|$")
axs[2].set_xlabel("Node i")

