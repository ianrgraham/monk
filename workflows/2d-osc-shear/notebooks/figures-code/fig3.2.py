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
m = "0.01"
zdx = 0
htls = ["634e14bc75836df75158fb15f379d8bb","1083a2a09eec2c6cf215ebc7a9706cac", "6976a7c10ada4159010403f8027f3ea5", "ecce68c50e28a33684826f28780bf6e9", "cfa1e4c0007fd65d020f809715a6f0b4", "9158e73b2e4c9fd1a565e1e5d1bc3af5"]
# file = f"/media/ian/Poppy/improved-nn/qle-v3/{htls[zdx]}/dataset_max-shear-{m}_period-1000.0_temp-1.9836e-03_frame-39.npz"
file = f"/media/ian/Poppy/improved-nn/qle-v3/{htls[zdx]}/dataset_init.npz"


data = np.load(file)
# data2 = np.load(file2)

# print(list(data.keys()))
N = 250
# norm = colors.LogNorm()
norm = colors.LogNorm()
norm2 = colors.LogNorm()
fig, axs_all = plt.subplots(2, 3, figsize=(15, 10))

axs = axs_all[0]
axs_bottom = axs_all[1]

plt.sca(axs[0])

def get_mode_participation(data):
    pos = data["pos"]
    evals = data["evals"]
    evecs = data["evecs"]

    c = np.linalg.norm(evecs, axis=2)
    # c = np.reshape(evecs, (evecs.shape[0], -1))
    # c = evecs
    # norm = np.linalg.norm(c, axis=(0,2))
    # for i in range(c.shape[0]):
    #     if evals[i] < 1e-5:
    #         c[i] = 0
    #     else:
    #         # print(evals[i])
    #         c[i] = c[i] / np.sqrt(evals[i])
    #         # c[i] = c[i] / np.linalg.norm(c[i])
    norm = np.zeros(c.shape[1])
    part = np.zeros(c.shape[1])
    for i in range(c.shape[0]):
        if evals[i] < 1e-5:
            continue
        term = np.sqrt(c[i]**2/evals[i])
        part += term**2
        norm += term

    return part/norm, norm

# print("done")

# plt.colorbar(sm, ax=axs)

overlap, normt = get_mode_participation(data)
pos = data["pos"]
axs[0].scatter(pos[:, 0], pos[:, 1], c=overlap, cmap='viridis', s=1, norm=norm)
axs_bottom[0].scatter(pos[:, 0], pos[:, 1], c=normt, cmap='viridis', s=1, norm=norm2)
bins = np.geomspace(1e-3, .5, 51)
# bins = np.linspace(0, .1, 101)
axs[2].hist(overlap, bins=bins, density=False, histtype='step', color='red')
axs_bottom[2].hist(normt, bins=bins, density=False, histtype='step', color='red')

esls = ["09d3dccc56f1fa9330b57901cd232ab7", "1d0fe2ba44a6f635f5504dcbcee0ebc9", "bf92a185d043ce3ef666e05e427c7e76", "a8f2c249ed85533b56b0729b7ab96d18", "c6d7afd0de62bfa16d05da47815227eb", "7ec2f6aca035e69077eab24383a21cb6"]
# file = f"/media/ian/Poppy/improved-nn/qle-v3/{esls[zdx]}/dataset_max-shear-{m}_period-1000.0_temp-1.9836e-03_frame-39.npz"
file = f"/media/ian/Poppy/improved-nn/qle-v3/{esls[zdx]}/dataset_init.npz"

data = np.load(file)

plt.sca(axs[1])

overlap, normt = get_mode_participation(data)
pos = data["pos"]
sm = axs[1].scatter(pos[:, 0], pos[:, 1], c=overlap, cmap='viridis', s=1, norm=norm)
sm2 = axs_bottom[1].scatter(pos[:, 0], pos[:, 1], c=normt, cmap='viridis', s=1, norm=norm2)
axs[2].hist(overlap, bins=bins, density=False, histtype='step', color='black')
axs_bottom[2].hist(normt, bins=bins, density=False, histtype='step', color='black')

plt.colorbar(sm, ax=axs)
plt.colorbar(sm2, ax=axs_bottom)
# plt.colorbar()
# axs[2].set_yscale('log')
axs[2].set_xscale('log')

plt.savefig("FIG3v2.png")