# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm, colors
import glob

from numba import njit, vectorize, float32

from typing import Callable, Optional, Union

import hoomd
import gsd.hoomd

import sys
import time
import pickle
import gc
import pathlib
import os

import signac

from scipy import optimize

# %%
from monk import nb, prep, pair, render, utils, grid, workflow
import freud

parent = pathlib.Path(os.getcwd()).parent / "config.yaml"
config = workflow.get_config(parent.as_posix())

# mpl.rcParams["text.usetex"] = True
# mpl.rcParams["text.latex.preamble"].append(r'\usepackage{xfrac}')

# %%
project: signac.Project = signac.get_project(root=config['root'])
project.doc

# %%
for job in project.find_jobs({"pot": "KA_WCA"}):
    files = sorted(glob.glob(job.fn("fine-equil/equil_*.gsd")))
    # print(files)

# %%
def vis_snap_scatter(snap: gsd.hoomd.Snapshot, ax: Optional[plt.Axes] = None, zoom=None, c=None):

    if ax is None:
        fig, ax = plt.subplots(dpi=150)
    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.set_xticks([])
    ax.set_yticks([])

    box = snap.configuration.box
    L = box[0]
    if zoom is not None:
        L /= zoom

    if c is None:
        ax.scatter(snap.particles.position[:, 0]/L, snap.particles.position[:, 1]/L, s=0.1)
    else:
        cmap = cm.jet
        norm = colors.Normalize(vmin=c.min(), vmax=c.max())
        ax.scatter(snap.particles.position[:, 0]/L, snap.particles.position[:, 1]/L, s=0.1, c=c, cmap=cmap, norm=norm)
        
    
    return ax


def vis_snap(snap: gsd.hoomd.Snapshot, ax: Optional[plt.Axes] = None, zoom=None, c=None):

    if ax is None:
        fig, ax = plt.subplots(dpi=150)
    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.set_xticks([])
    ax.set_yticks([])

    box = snap.configuration.box
    L = box[0]
    if zoom is not None:
        L /= zoom

    if c is None:
        ax.scatter(snap.particles.position[:, 0]/L, snap.particles.position[:, 1]/L, s=0.1)
    else:
        cmap = cm.jet
        norm = colors.Normalize(vmin=1e-1, vmax=1.21)
        ax.scatter(snap.particles.position[:, 0]/L, snap.particles.position[:, 1]/L, s=0.1, c=c, cmap=cmap, norm=norm)
        
    
    return fig, ax


# %%
print(len(files))
idx = 0
for file in files[-1:]:
    print(file)
    temp = utils.extract_between(file, "temp-", ".gsd")
    traj = gsd.hoomd.open(file)
    for jdx in range(200):
        snap = traj[jdx]
        snap2 = traj[jdx + 100]
        box = freud.box.Box.from_box(snap.configuration.box)


        # %%
        diff = box.wrap(snap2.particles.position - snap.particles.position)
        diff = np.linalg.norm(diff, axis=1)*7.14
        diff = 1 - np.sin(diff)/diff

        # print(diff.min(), diff.max(), diff.mean())

        # %%
        fig, ax = vis_snap(snap, zoom=2, c=diff)

        # %%
        # len(traj)

        fig.savefig(f"dyn-hetero-movie/{idx:04d}-wca-{temp}.png", dpi=200, bbox_inches='tight')
        idx += 1

        del fig, ax

# %%



