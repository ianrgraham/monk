# %% [markdown]
# # Analyze AQS stress soft corr
# 
# We have AQS simulation (with and without preshear) at a number of oscillatory
# shears
# 
# We need to compute softness intercycle and get the local virial intercycle

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import polars as pl
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
import shutil

from scipy import stats
import signac
import freud
import gsd.hoomd
import hoomd
import schmeud
import schmeud._schmeud as schmeud_rs
from schmeud._schmeud import statics
from schmeud._schmeud import locality
from schmeud import ml
from tqdm import tqdm

from dataclasses import dataclass
from collections import defaultdict

from numba import njit

# %%
from monk import nb, prep as prep_mod, pair, render, utils, workflow, grid

parent = pathlib.Path(os.getcwd()).parent.parent / "config.yaml"
config = workflow.get_config(parent.as_posix())
parent, config

# %%
project: signac.Project = signac.get_project(root=config['root'])
project.doc

# %% [markdown]
# ## Generate data

# %%
@dataclass(frozen=True, eq=True)
class Statepoint:
    max_shear: float
    period: float
    temp: float
    prep: str

# %%
with open("../sklearn/svc.pkl", "rb") as f:
    pipe = pickle.load(f)

with open("../sklearn/svc_type0.pkl", "rb") as f:
    pipe0 = pickle.load(f)

with open("../sklearn/svc_type1.pkl", "rb") as f:
    pipe1 = pickle.load(f)

# %%
class InteractionComputer:

    def _setup_nve_sim(self, pot, dt=0.01):
        if isinstance(pot, str):
            pot = [pot]
        elif isinstance(pot, list):
            pass
        else:
            raise ValueError("The 'potential' must be a string or list")

        integrator = hoomd.md.Integrator(dt=dt)
        pair_func, args = prep_mod.search_for_pair(pot)
        tree = hoomd.md.nlist.Tree(0.3)
        print("Using pair function: ", pair_func)
        force = pair_func(tree, *args)

        nvt = hoomd.md.methods.NVE(hoomd.filter.All())
        integrator.forces = [force]
        integrator.methods = [nvt]

        self._sim.operations.integrator = integrator

        self._nlist = tree

        return nvt

    def __init__(self, filename, pot, frame=0, device=None, seed=None):
        if device is None:
            device = hoomd.device.CPU()
        sim = hoomd.Simulation(device, seed=seed)
        sim.create_state_from_gsd(filename, frame=frame)
        self._sim = sim
        self._file_handle = gsd.hoomd.open(name=filename, mode='rb')

        self._setup_nve_sim(pot)

        self._sim.always_compute_pressure = True

        self._snap = None

        sim.run(0)

    def get_forces(self):
        assert len(self._sim.operations.integrator.forces) == 1
        self._sim.run(0)
        return self._sim.operations.integrator.forces[0].forces
    
    def get_virials(self):
        assert len(self._sim.operations.integrator.forces) == 1
        self._sim.run(0)
        return self._sim.operations.integrator.forces[0].virials
    
    def get_pairs(self):
        self._sim.run(0)
        return self._nlist.pair_list()
    
    def get_snapshot(self):
        if self._snap is None or self._sim.timestep != self._snap.configuration.step:
            self._snap = self._sim.state.get_snapshot()
        return self._snap

    def get_positions(self):
        snap = self.get_snapshot()
        return snap.positions
    
    def get_box(self):
        return self._sim.state.box

    def set_frame(self, frame):
        snap = self._file_handle[frame]
        snap = hoomd.Snapshot.from_gsd_snapshot(snap, hoomd.communicator.Communicator())
        self._sim.state.set_snapshot(snap)

    def set_file(self, filename, frame=0):
        self._file_handle = gsd.hoomd.open(name=filename, mode='rb')
        self.set_frame(frame)

# %%

# inter_computer = None

for job in project:
    print(job)
    prep = job.sp["prep"]
    
    experiments = sorted(glob.glob(job.fn("aqs_experiments_preshear/*/*/traj_period-*.gsd")))
    # if inter_computer is None:
    inter_computer = InteractionComputer(experiments[0], job.sp["pot"], frame=0)
    if len(experiments) == 0:
        continue
    for exper in experiments:
        max_shear = utils.extract_between(exper, "max-shear-", "/")
        period = utils.extract_between(exper, "period-", "_preshear")
        temp = utils.extract_between(exper, "temp-", "/")
        df_path = exper.replace("traj_period", "soft-stress_period").replace(".gsd", ".parquet")
        sp = Statepoint(max_shear=float(max_shear), period=float(period), temp=float(temp), prep=prep)

        if job.isfile(df_path):
            # lazily get df
            df = pl.read_parquet(job.fn(df_path), use_pyarrow=True)
            # check if "stress" column exists
            if "sigma_xx" in df.columns:
                print("exists, data is good, passing")
                continue
            else:
                print("exists, but data is bad")
                
        #     dataset = pl.read_parquet(job.fn(df_path), use_pyarrow=True)
        # else:
        #     continue

        inter_computer.set_file(exper, frame=0)

        traj = gsd.hoomd.open(exper, "rb")

        frames = []
        xys = []
        types = []
        softness = []
        stresses = []


        for frame in range(len(traj)):
            frames.append(frame)
            snap = traj[frame]
            box = snap.configuration.box
            # square_box = np.array([box[0], box[1], 0, 0, 0, 0])
            # freud_box = freud.box.Box.from_box(square_box)
            # pos = freud_box.wrap(snap.particles.position)

            xys.append(box[3])

            inter_computer.set_frame(frame)
            virials = inter_computer.get_virials()
            #virial_trace = virials[:, 0] + virials[:, 3]
            # virial_trace = virials[:, 1]
            # xx = virials[:, 0]
            # yy = virials[:, 3]
            # xy = virials[:, 1]
            # diff = xx - yy
            # virial_trace = np.sqrt(xy**2 + diff**2)

            typeids = snap.particles.typeid
            types.append(typeids)
            softb = np.zeros_like(typeids, dtype=np.float32)

            query_indices0 = np.arange(snap.particles.N)[typeids == 0]
            sfs0 = ml.compute_structure_functions_snap(snap, query_indices0)
            soft0 = pipe0.decision_function(sfs0)

            query_indices1 = np.arange(snap.particles.N)[typeids == 1]
            sfs1 = ml.compute_structure_functions_snap(snap, query_indices1)
            soft1 = pipe1.decision_function(sfs1)

            softb[typeids == 0] = soft0
            softb[typeids == 1] = soft1

            softness.append(softb)
            stresses.append(virials)

        stresses2 = np.array(stresses, dtype=np.float32)
        dataset = pl.DataFrame({"frame": frames, "xy": xys, "id": np.array(types), "soft": np.array(softness), "sigma_xx": stresses2[:, :, 0], "sigma_yy": stresses2[:, :, 3], "sigma_xy": stresses2[:, :, 1]})
        # dataset = dataset.explode(["id", "soft"]).reset_index(drop=True)
        dataset.write_parquet(job.fn(df_path), use_pyarrow=True)
    del inter_computer
    #     break
    # break

# %% [markdown]
# ## Analyze data

# %%



