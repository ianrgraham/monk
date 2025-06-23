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


x = np.linspace(-1/2*np.pi, (2.5)*np.pi, 100)
y = np.sin(x)
# axin = axs[0].inset_axes([0.35, 0.65, 0.3, 0.3])
fig, axs = plt.subplots(1, 2, figsize=(6, 2), dpi=300)
axin = axs[0]
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
axin.set_title(r"$D^2_{min;strob}$")
# set background color
# axin.patch.set_facecolor('gray')

axin = axs[1]
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
axin.set_title(r"$D^2_{min;ptp}$")


# plt.tight_layout()

plt.savefig("FIG0.png")