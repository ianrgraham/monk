import numpy as np
import pandas as pd
import freud
import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib import colors, cm

import os
import sys
import glob
# 09d3dccc56f1fa9330b57901cd232ab7
# 634e14bc75836df75158fb15f379d8bb
file = "/media/ian/Poppy/improved-nn/qle-v3/634e14bc75836df75158fb15f379d8bb/dataset_max-shear-0.12_period-1000.0_temp-1.9836e-03_frame-39.npz"
file2 = "/media/ian/Poppy/improved-nn/qle-v3/09d3dccc56f1fa9330b57901cd232ab7/dataset_max-shear-0.12_period-1000.0_temp-1.9836e-03_frame-39.npz"

data = np.load(file)
data2 = np.load(file2)

# print(list(data.keys()))

pos = data["pos"]
evals = data["evals"]
evecs = data["evecs"]
orig_evecs = data["orig_evecs"]

# compute participation ratios
part_norm = np.linalg.norm(evecs, axis=2)
vec_norm = np.linalg.norm(evecs, axis=(1, 2))

# pr = np.sum(part_norm**2, axis=1)/evals

# plt.plot(pr[3:])
# plt.yscale('log')
idx = 18


# c = np.linalg.norm(orig_evecs[idx], axis=1)**2/evals[idx]
c = np.linalg.norm(orig_evecs[idx], axis=1)

c = np.zeros_like(c)

for i in range(idx):
    if evals[i] < 1e-5:
        continue
    c += np.linalg.norm(orig_evecs[i], axis=1)**2/evals[i]

fig, ax = plt.subplots()

# ax.set_aspect('equal')

vmax = np.quantile(c, 0.999)
vmin = np.quantile(c, 0.10)
# norm = colors.LogNorm(vmin, vmax)
norm = colors.Normalize(vmin, vmax)

# bins = np.geomspace(np.min(c), np.max(c), 100)
# plt.hist(c, bins=bins, density=True, histtype='step', color='black')
# plt.xscale('log')

# plt.xlim(-80, -50)
# plt.ylim(-50, -20)

plt.scatter(pos[:, 0], pos[:, 1], c=c, cmap='viridis', s=2, norm=norm)
plt.colorbar()

# add (a) and (b) labels
plt.text(-0.05, 1.05, r"a", transform=ax.transAxes, weight="bold", size=15)

plt.savefig("FIG3.png")