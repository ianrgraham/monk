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
import gsd.hoomd
import freud
import schmeud
from schmeud._schmeud import dynamics as schmeud_dynamics
from schmeud._schmeud import locality as schmeud_locality

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

import gudhi

# %%
for delta, jobs_iter in project.find_jobs({"delta": 0.0}, {"_CRYSTAL": {"$exists": False}}).groupby("delta"):
    for job in jobs_iter:
        for run_dir in sorted(glob.glob(job.fn("short_runs/temp-*"))):
            run_dir = pathlib.Path(run_dir)
            # get softness df
            softness_df = pd.read_parquet(run_dir / "struct-descr.parquet")
            # get traj
            traj = gsd.hoomd.open(run_dir / "traj.gsd")


            break
        break

# %%
soft = softness_df[softness_df["frame"] == 0]["softness"].values.astype(np.float32)

# %%
soft.dtype

# %%
l = traj[0].configuration.box[0]

# %%
points = traj[0].particles.position
points = points[traj[0].particles.typeid == 0]

# %%
grid = schmeud_locality.particle_to_grid_cube(points, soft, l, 10)

# %%
# gudhi.CubicalComplex.


