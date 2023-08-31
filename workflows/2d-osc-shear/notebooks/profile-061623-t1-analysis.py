# %%
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

# %%
from monk import nb, prep, pair, render, utils, workflow, grid

parent = pathlib.Path(os.getcwd()).parent / "config.yaml"
config = workflow.get_config(parent.as_posix())
parent, config

# %%
project: signac.Project = signac.get_project(root=config['root'])
project.doc

# %%
@dataclass(frozen=True, eq=True)
class Statepoint:
    max_shear: float
    period: float
    temp: float

# %%
# get t1 events
# reversible and irreversible rearrangments

rev_quant = defaultdict(list)

for job in project:
    print(job)

    experiments = sorted(glob.glob(job.fn("experiments/*/*/*")))

    for exper in experiments:
        max_shear = utils.extract_between(exper, "max-shear-", "/")
        period = utils.extract_between(exper, "period-", ".gsd")
        temp = utils.extract_between(exper, "temp-", "/")
        sp = Statepoint(max_shear=float(max_shear), period=float(period), temp=float(temp))
        
        if float(period) != 1000.0:
            continue

        traj = gsd.hoomd.open(exper)

        print(max_shear, period, temp)

        rev_count = []
        irr_count = []
        voro = freud.locality.Voronoi()
        for i in range(1, 20):
            print(i)

            rearranged = set()

            snap_0 = traj[-1 + i*40] # initial state
            snap_1 = traj[9 + i*40] # peak
            snap_2 = traj[19 + i*40] # back to zero
            snap_3 = traj[29 + i*40] # min peak
            snap_4 = traj[-1 + (i+1)*40] # full cycle complete

            box_0 = snap_0.configuration.box[:]
            box_1 = snap_1.configuration.box[:]
            box_2 = snap_2.configuration.box[:]
            box_3 = snap_3.configuration.box[:]
            box_4 = snap_4.configuration.box[:]

            voro.compute((box_0, snap_0.particles.position)) # this is very costly
            nlist = voro.nlist

            # this is also very costly
            neighbors = set([frozenset(set([i, j])) for i, j in zip(nlist.query_point_indices, nlist.point_indices)])

            next_to_process = zip([box_1, box_2, box_3, box_4], [snap_1, snap_2, snap_3, snap_4])

            for box, snap in next_to_process:
                voro.compute((box, snap.particles.position))
                nlist = voro.nlist
                neighbors_ = set([frozenset(set([i, j])) for i, j in zip(nlist.query_point_indices, nlist.point_indices)])
                print(len(neighbors_))
                rearranged |= neighbors - neighbors_
            rev = rearranged & neighbors_
            irr = rearranged - rev
            rev_count.append(len(rev))
            irr_count.append(len(irr))
            break
        break
    break

# save to pickle
# with open(project.fn("results/rev-quant.pkl"), "wb") as f:
#     pickle.dump(rev_quant, f)

# %%
len(neighbors)/(3*32768 - 1), len(neighbors_), 3*32768 - 1

# %%
2**15

# %%
plt.plot(np.array(rev_count)/len(neighbors_))
plt.plot(np.array(irr_count)/len(neighbors_))

# %%
len(rearranged)

# %%
rearranged.intersection(neighbors_)

# %%



