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
m = "0.02"
zdx = 4
htls = ["634e14bc75836df75158fb15f379d8bb","1083a2a09eec2c6cf215ebc7a9706cac", "6976a7c10ada4159010403f8027f3ea5", "ecce68c50e28a33684826f28780bf6e9", "cfa1e4c0007fd65d020f809715a6f0b4", "9158e73b2e4c9fd1a565e1e5d1bc3af5"]
# file = f"/media/ian/Poppy/improved-nn/qle-v3/634e14bc75836df75158fb15f379d8bb/dataset_max-shear-{m}_period-1000.0_temp-1.9836e-03_frame-39.npz"
file = f"/media/ian/Poppy/improved-nn/qle-v3/{htls[zdx]}/dataset_init.npz"


data = np.load(file)
# data2 = np.load(file2)

# print(list(data.keys()))
N = 1000
# norm = colors.LogNorm()
norm = colors.Normalize(vmax=20)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

plt.sca(axs[0])

def get_overlap(data, mat=True):
    pos = data["pos"]
    evals = data["evals"]
    evecs = data["evecs"]

    # c = np.linalg.norm(evecs, axis=2)
    c = evecs
    for i in range(c.shape[0]):
        if evals[i] < 1e-5:
            c[i] = 0
        else:
            # print(evals[i])
            c[i] = c[i] / np.sqrt(evals[i])
            # c[i] = c[i] / np.linalg.norm(c[i])
    if mat:
        overlap = np.zeros((N, N))
        # print("hey")
        for i in range(N):
            if evals[i] < 1e-5:
                continue
            for j in range(N):
                if evals[j] < 1e-5:
                    continue
                if i != j:
                    overlap[i, j] = np.linalg.norm(c[i] * c[j])
        return overlap
    else:
        overlap = []
        for i in range(N):
            if evals[i] < 1e-5:
                continue
            for j in range(i+1, N):
                if evals[j] < 1e-5:
                    continue
                overlap.append(np.linalg.norm(c[i] * c[j]))
        return np.array(overlap)

# print("done")

overlap = get_overlap(data)
sm = axs[0].matshow(overlap, norm=norm)

# plt.colorbar(sm, ax=axs)

overlap = get_overlap(data, mat=False)
bins = np.geomspace(1e-3, 40, 101)
# bins = np.linspace(0, 10, 61)
axs[2].hist(overlap, bins=bins, density=True, histtype='step', color='red')

esls = ["09d3dccc56f1fa9330b57901cd232ab7", "1d0fe2ba44a6f635f5504dcbcee0ebc9", "bf92a185d043ce3ef666e05e427c7e76", "a8f2c249ed85533b56b0729b7ab96d18", "c6d7afd0de62bfa16d05da47815227eb", "7ec2f6aca035e69077eab24383a21cb6"]
# file = f"/media/ian/Poppy/improved-nn/qle-v3/09d3dccc56f1fa9330b57901cd232ab7/dataset_max-shear-{m}_period-1000.0_temp-1.9836e-03_frame-39.npz"
file = f"/media/ian/Poppy/improved-nn/qle-v3/{esls[zdx]}/dataset_init.npz"

data = np.load(file)

plt.sca(axs[1])

overlap = get_overlap(data)
sm = axs[1].matshow(overlap, norm=norm)

overlap = get_overlap(data, mat=False)
axs[2].hist(overlap, bins=bins, density=True, histtype='step', color='black')

plt.colorbar(sm, ax=axs)
# plt.colorbar()
axs[2].set_yscale('log')
axs[2].set_xscale('log')

plt.savefig("FIG3v1.png")