#!/usr/bin/env python
# coding: utf-8

# # Imports and project setup

# In[6]:


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


# ## Function definitions

# In[2]:


def delta_to_phi(delta):
    return 1.2 - delta*0.2

def make_axes_label(ax, label, pos=(0.0, 1.05), fontsize=None):
    ax.text(*pos, label, fontsize=fontsize, transform=ax.transAxes)


# In[3]:


def compute_unwrapped_pos_from_deltas(traj):
    ref_snap = traj[0]
    ref_pos = ref_snap.particles.position.copy()
    box = freud.box.Box.from_box(ref_snap.configuration.box)
    pos_shape = ref_pos.shape
    pos = np.zeros((len(traj), *pos_shape), dtype=np.float32)
    pos[0] = ref_pos
    for i, snap in enumerate(traj[1:]):
        next_pos = snap.particles.position.copy()
        pos[i+1] = box.wrap(next_pos - ref_pos) + pos[i]
        ref_pos = next_pos

    return pos

def extract_relative_timesteps(traj):
    
    tsteps = np.zeros(len(traj))
    tstep0 = traj[0].log['Simulation/timestep'][0]
    for i in range(len(traj)):
        tsteps[i] = traj[i].log['Simulation/timestep'][0] - tstep0

    return tsteps


def sisf(pos, k=7.14):

    term = k*np.linalg.norm(pos - pos[0], axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.mean(np.nan_to_num(np.sin(term)/term, nan=1.0), axis=1)



def is_file_older_than(file, cutoff):
    mtime = datetime.utcfromtimestamp(os.path.getmtime(file))
    if mtime < cutoff:
        return True
    return False


# In[4]:


def digitize_mask(x, mu_min, mu_max, bins):
    digs = np.floor((x - mu_min) / (mu_max - mu_min) * bins).astype(np.int32)
    return np.ma.masked_array(digs, mask=(digs < 0) | (digs >= bins))


def digitize_in_bounds(x, mu_min, mu_max, bins):
    digs = digitize_mask(x, mu_min, mu_max, bins)
    return digs[~digs.mask].data


def bin_linspaced_data(x, mu_min, mu_max, bins):
    digs = digitize_in_bounds(x, mu_min, mu_max, bins)
    return np.bincount(digs, minlength=bins)


def pair_excess_entropy_binary(g_a, g_b, f_a, f_b, r, phi):
    dr = r[1] - r[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return -2.0*np.pi*phi*f_a*np.sum(np.nan_to_num(g_a*np.log(g_a) - g_a + 1)*r*r)*dr \
                + -2.0*np.pi*phi*f_b*np.sum(np.nan_to_num(g_b*np.log(g_b) - g_b + 1)*r*r)*dr


# In[5]:


@njit
def excess_entropy(r, g, dr):
    prelim = (g*np.log(g)-g+1)*r*r*dr
    prelim[np.isnan(prelim)] = 0
    return -2.0*np.pi*np.sum(prelim)

@njit
def local_excess_entropy(r, sfs, dr):
    vol = 4/3*np.pi*(r[-1]**3)
    inv_shell = 1/(4*np.pi*(r**2)*dr)
    out = np.zeros(len(sfs))
    for i in range(len(sfs)):
        sf = sfs[i,::2] + sfs[i,1::2]
        inv_n = 1/np.sum(sf)
        g = sf*inv_shell*inv_n*vol
        out[i] = excess_entropy(r, g, dr)
    return out


@njit
def local_excess_entropy_binary(r, sfs, dr):
    vol = 4/3*np.pi*(r[-1]**3)
    inv_shell = 1/(4*np.pi*(r**2)*dr)
    out = np.zeros(len(sfs))
    for i in range(len(sfs)):
        na = np.sum(sfs[i,::2])
        nb = np.sum(sfs[i,1::2])
        n = na + nb
        inv_na = 1/na
        inv_nb = 1/nb
        g_a = sfs[i,::2]*inv_shell*inv_na*vol
        g_b = sfs[i,1::2]*inv_shell*inv_nb*vol
        out[i] = na/n*excess_entropy(r, g_a, dr) + nb/n*excess_entropy(r, g_b, dr)
    return out


# In[6]:


@njit
def local_excess_entropy_density(r, sfs, dr):
    vol = 4/3*np.pi*(r[-1]**3)
    inv_shell = 1/(4*np.pi*(r**2)*dr)
    out = np.zeros(len(sfs))
    for i in range(len(sfs)):
        sf = sfs[i,::2] + sfs[i,1::2]
        inv_n = 1/np.sum(sf)
        g = sf*inv_shell*inv_n*vol
        out[i] = excess_entropy(r, g, dr)/vol/inv_n
    return out


@njit
def local_excess_entropy_binary_density(r, sfs, dr):
    vol = 4/3*np.pi*(r[-1]**3)
    inv_shell = 1/(4*np.pi*(r**2)*dr)
    out = np.zeros(len(sfs))
    for i in range(len(sfs)):
        na = np.sum(sfs[i,::2])
        nb = np.sum(sfs[i,1::2])
        n = na + nb
        inv_na = 1/na
        inv_nb = 1/nb
        g_a = sfs[i,::2]*inv_shell*inv_na*vol
        g_b = sfs[i,1::2]*inv_shell*inv_nb*vol
        out[i] = (na/n*excess_entropy(r, g_a, dr) + nb/n*excess_entropy(r, g_b, dr))*n/vol
    return out
@njit
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

@njit
def local_excess_entropy_binary_density_var(r, sfs, dr, alpha=0.5, taper=(2.0, 1.0)):
    if not (alpha <= 1.0 and alpha >= 0.0):
        raise ValueError("alpha must be between 0 and 1")
    alpha_n = 1 - alpha
    vol = 4/3*np.pi*(r[-1]**3)
    inv_shell = 1/(4*np.pi*(r**2)*dr)
    out = np.zeros(len(sfs))
    log_taper = sigmoid((r - taper[0])/taper[1])
    nlt = 1 - log_taper
    for i in range(len(sfs)):
        na = np.sum(sfs[i,::2])
        nb = np.sum(sfs[i,1::2])
        n = na + nb
        inv_na = 1/na
        inv_nb = 1/nb
        g_a = sfs[i,::2]*inv_shell*inv_na*vol*nlt + log_taper
        g_b = sfs[i,1::2]*inv_shell*inv_nb*vol*nlt + log_taper
        out[i] = (alpha*na/n*excess_entropy(r, g_a, dr) + alpha_n*nb/n*excess_entropy(r, g_b, dr))*n/vol
    return out


@njit
def local_excess_entropy_binary_density_var_exp(r, sfs, dr, alpha=0.5, taper=(2.0, 0.1), end=-1):
    if not (alpha <= 1.0 and alpha >= 0.0):
        raise ValueError("alpha must be between 0 and 1")
    alpha_n = 1 - alpha
    vol = 4/3*np.pi*(r[end]**3)
    inv_shell = 1/(4*np.pi*(r**2)*dr)
    out = np.zeros(len(sfs))
    log_taper = sigmoid((r - taper[0])/taper[1])
    nlt = 1 - log_taper
    for i in range(len(sfs)):
        na = np.sum(sfs[i,::2][:end])
        nb = np.sum(sfs[i,1::2][:end])
        n = na + nb
        inv_na = 1/na
        inv_nb = 1/nb
        g_a = sfs[i,::2]*inv_shell*inv_na*vol*nlt + log_taper
        g_b = sfs[i,1::2]*inv_shell*inv_nb*vol*nlt + log_taper
        out[i] = (alpha*na/n*excess_entropy(r, g_a, dr) + alpha_n*nb/n*excess_entropy(r, g_b, dr))*n/vol
    return out

# # Figure 6 - Final energy and entropy fits

# In[10]:


fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
cmap = cm.jet
norm = colors.Normalize(0.0, 0.4)

for delta in [0.0, 0.1, 0.2, 0.3, 0.4]:
    file = project.fn(f"prob_rearrang/rdfs-bin-softness-fire_delta-{delta}.pickle")

    data = pickle.load(open(file, "rb"))

    itemps = data["inv_t"]
    in_cuts = data["data"][0]["cuts"]
    f_a = 0.8
    f_b = 0.2
    r = data["r_bc"]
    phi = data["phi"]
    fits = []
    entropy = []
    for i in range(len(data["cuts"])):
        rearrang = [d["rearrang"][i] for d in data["data"]]
        g_a = data["data"][3]["rdfs"][in_cuts[i]]["g_a"]
        g_b = data["data"][3]["rdfs"][in_cuts[i]]["g_b"]
        s = pair_excess_entropy_binary(g_a, g_b, f_a, f_b, r, phi)
        entropy.append(s)
        p = np.polyfit(itemps, np.log(rearrang), 1)

        fits.append(p)

    soft = pickle.load(open(project.fn(f"prob_rearrang/fit_softness_delta-{delta}.pickle"), "rb"))

    def reject_outliers(sr, iq_range=0.8):
        pcnt = (1 - iq_range) / 2
        qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
        iqr = qhigh - qlow
        return sr[ (sr - median).abs() <= iqr]

    data1 = []
    keys = list(soft["data"].keys())
    for i in range(len(keys)-1):
        v1 = soft["data"][keys[i]]["p"]
        for j in range(i+1, len(keys)):
            v2 = soft["data"][keys[j]]["p"]
            data1.append(1/((v2[1] - v1[1]) / (v1[0] - v2[0])))
    T_a = np.mean(reject_outliers(pd.Series(data1)).to_numpy())

    fits = np.array(fits)

    # axs[0].plot(entropy, fits[:,1] + fits[:,0]/T_a)

    axs[0].plot(entropy, fits[:,1]/(1.0 - delta/2.0**(1/6)), ".", linestyle="", label=f"{delta:.1f}", color=cmap(norm(delta)))
    axs[1].plot(entropy, -fits[:,0]/(1.0 - delta/2.0**(1/6))/T_a, ".", linestyle="", label=f"{delta:.1f}", color=cmap(norm(delta)))

axs[0].set_xlabel(r"$s^{(2)}$", size="xx-large")
axs[0].set_ylabel(r"$\frac{\Sigma}{\sigma'}$", size="xx-large")
plt.sca(axs[0])
plt.legend(title=r"$\Delta$")
# make_axes_label(axs[0], "(a)")

axs[1].set_xlabel(r"$s^{(2)}$", size="xx-large")
axs[1].set_ylabel(r"$\frac{\Delta E}{\sigma' \cdot T_A }$", size="xx-large")
plt.sca(axs[1])
plt.legend(title=r"$\Delta$")
# make_axes_label(axs[1], "(b)")

# plt.savefig("fig6-fixed.png", dpi=200)


# In[7]:


fig, axs = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
cmap = cm.jet
norm = colors.Normalize(0.0, 0.4)

for delta in [0.0, 0.1, 0.2, 0.3, 0.4]:
    file = project.fn(f"prob_rearrang/rdfs-bin-softness-fire_delta-{delta}.pickle")

    data = pickle.load(open(file, "rb"))

    itemps = data["inv_t"]
    in_cuts = data["data"][0]["cuts"]
    f_a = 0.8
    f_b = 0.2
    r = data["r_bc"]
    phi = data["phi"]
    fits = []
    entropy = []
    for i in range(len(data["cuts"])):
        rearrang = [d["rearrang"][i] for d in data["data"]]
        g_a = data["data"][3]["rdfs"][in_cuts[i]]["g_a"]
        g_b = data["data"][3]["rdfs"][in_cuts[i]]["g_b"]
        s = pair_excess_entropy_binary(g_a, g_b, f_a, f_b, r, phi)
        entropy.append(s)
        p = np.polyfit(itemps, np.log(rearrang), 1)

        fits.append(p)

    soft = pickle.load(open(project.fn(f"prob_rearrang/fit_softness_delta-{delta}.pickle"), "rb"))

    def reject_outliers(sr, iq_range=0.8):
        pcnt = (1 - iq_range) / 2
        qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
        iqr = qhigh - qlow
        cond = (sr - median).abs() <= iqr
        return cond

    data1 = []
    keys = list(soft["data"].keys())
    for i in range(len(keys)-1):
        v1 = soft["data"][keys[i]]["p"]
        for j in range(i+1, len(keys)):
            v2 = soft["data"][keys[j]]["p"]
            data1.append(1/((v2[1] - v1[1]) / (v1[0] - v2[0])))
    ser_data = pd.Series(data1)
    cond = reject_outliers(ser_data)
    T_a = np.mean(ser_data[cond].to_numpy())
    P_rs = []
    for i in range(len(keys)):
        v1 = soft["data"][keys[i]]["p"]
        P_rs.append(np.exp(v1[1] - v1[0]/T_a))
    P_r = np.log(np.mean(P_rs))
    print(P_r)

    fits = np.array(fits)

    print(delta, T_a, P_r)

    axs[0].plot(entropy, fits[:,1]/P_r, ".", linestyle="", label=f"{delta:.1f}", color=cmap(norm(delta)))
    axs[1].plot(entropy, -fits[:,0]/P_r/T_a, ".", linestyle="", label=f"{delta:.1f}", color=cmap(norm(delta)))

axs[0].set_xlabel(r"$s^{(2)}$")
axs[0].set_ylabel(r"$\frac{\Sigma T_A}{\Delta F(T_A)}$")
plt.sca(axs[0])
plt.legend(title=r"$\Delta$")
make_axes_label(axs[0], "(a)")

axs[1].set_xlabel(r"$s^{(2)}$")
axs[1].set_ylabel(r"$\frac{\Delta E}{\Delta F(T_A) }$")
plt.sca(axs[1])
# plt.legend(title=r"$\Delta$")
make_axes_label(axs[1], "(b)")

plt.savefig("fig6-pres.png", dpi=200)


# In[15]:


fig, axs = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
cmap = cm.jet
norm = colors.Normalize(0.0, 0.4)

for delta in [0.0, 0.1, 0.2, 0.3, 0.4]:
    file = project.fn(f"prob_rearrang/rdfs-bin-softness-fire_delta-{delta}.pickle")

    data = pickle.load(open(file, "rb"))

    itemps = data["inv_t"]
    in_cuts = data["data"][0]["cuts"]
    f_a = 0.8
    f_b = 0.2
    r = data["r_bc"]
    phi = data["phi"]
    fits = []
    entropy = []
    for i in range(len(data["cuts"])):
        rearrang = [d["rearrang"][i] for d in data["data"]]
        g_a = data["data"][3]["rdfs"][in_cuts[i]]["g_a"]
        g_b = data["data"][3]["rdfs"][in_cuts[i]]["g_b"]
        s = pair_excess_entropy_binary(g_a, g_b, f_a, f_b, r, phi)
        entropy.append(s)
        p = np.polyfit(itemps, np.log(rearrang), 1)

        fits.append(p)

    soft = pickle.load(open(project.fn(f"prob_rearrang/fit_softness_delta-{delta}.pickle"), "rb"))

    def reject_outliers(sr, iq_range=0.8):
        pcnt = (1 - iq_range) / 2
        qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
        iqr = qhigh - qlow
        cond = (sr - median).abs() <= iqr
        return cond

    data1 = []
    keys = list(soft["data"].keys())
    for i in range(len(keys)-1):
        v1 = soft["data"][keys[i]]["p"]
        for j in range(i+1, len(keys)):
            v2 = soft["data"][keys[j]]["p"]
            data1.append(1/((v2[1] - v1[1]) / (v1[0] - v2[0])))
    ser_data = pd.Series(data1)
    cond = reject_outliers(ser_data)
    T_a = np.mean(ser_data[cond].to_numpy())
    P_rs = []
    for i in range(len(keys)):
        v1 = soft["data"][keys[i]]["p"]
        P_rs.append(np.exp(v1[1] - v1[0]/T_a))
    P_r = np.log(np.mean(P_rs))
    print(P_r)

    fits = np.array(fits)

    print(delta, T_a, P_r)

    axs[0].plot(entropy, fits[:,1]/P_r, ".", linestyle="", label=f"{delta:.1f}", color=cmap(norm(delta)))
    axs[1].plot(entropy, -fits[:,0]/P_r/T_a, ".", linestyle="", label=f"{delta:.1f}", color=cmap(norm(delta)))

axs[0].set_xlabel(r"$s^{(2)}$")
axs[0].set_ylabel(r"$\frac{\Sigma T_A}{\Delta F(T_A)}$")
plt.sca(axs[0])
plt.legend(title=r"$\Delta$")
make_axes_label(axs[0], "(a)")

axs[1].set_xlabel(r"$s^{(2)}$")
axs[1].set_ylabel(r"$\frac{\Delta E}{\Delta F(T_A) }$")
plt.sca(axs[1])
# plt.legend(title=r"$\Delta$")
make_axes_label(axs[1], "(b)")

plt.savefig("fig6-pres.png", dpi=200)


# In[84]:


fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
cmap = cm.jet
norm = colors.Normalize(0.0, 0.4)
force_data = np.save(project.fn("structure/cage-force-sup.npy"))

for delta in [0.0, 0.1, 0.2, 0.3, 0.4]:
    file = project.fn(f"prob_rearrang/rdfs-bin-softness-fire_delta-{delta}.pickle")

    data = pickle.load(open(file, "rb"))

    itemps = data["inv_t"]
    in_cuts = data["data"][0]["cuts"]
    f_a = 0.8
    f_b = 0.2
    r = data["r_bc"]
    phi = data["phi"]
    fits = []
    entropy = []
    for i in range(len(data["cuts"])):
        rearrang = [d["rearrang"][i] for d in data["data"]]
        g_a = data["data"][3]["rdfs"][in_cuts[i]]["g_a"]
        g_b = data["data"][3]["rdfs"][in_cuts[i]]["g_b"]
        s = pair_excess_entropy_binary(g_a, g_b, f_a, f_b, r, phi)
        entropy.append(s)
        p = np.polyfit(itemps, np.log(rearrang), 1)

        fits.append(p)

    soft = pickle.load(open(project.fn(f"prob_rearrang/fit_softness_delta-{delta}.pickle"), "rb"))

    def reject_outliers(sr, iq_range=0.8):
        pcnt = (1 - iq_range) / 2
        qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
        iqr = qhigh - qlow
        return sr[ (sr - median).abs() <= iqr]

    data1 = []
    keys = list(soft["data"].keys())
    for i in range(len(keys)-1):
        v1 = soft["data"][keys[i]]["p"]
        for j in range(i+1, len(keys)):
            v2 = soft["data"][keys[j]]["p"]
            data1.append((v2[1] - v1[1]) / (v1[0] - v2[0]))
    T_a = np.mean(reject_outliers(pd.Series(data1)).to_numpy())

    fits = np.array(fits)

    axs[0].plot(entropy, fits[:,1]/(1.2/phi)**(1/3)/(1.0 - delta/2.0**(1/6)), ".", linestyle="", label=f"{delta:.1f}", color=cmap(norm(delta)))
    axs[1].plot(entropy, -fits[:,0]*T_a/(1.2/phi)**(1/3)/(1.0 - delta/2.0**(1/6)), ".", linestyle="", label=f"{delta:.1f}", color=cmap(norm(delta)))

axs[0].set_xlabel(r"$s^{(2)}$")
axs[0].set_ylabel(r"$\frac{\Delta E}{F^*}$")
plt.sca(axs[0])
plt.legend(title=r"$\Delta$")
make_axes_label(axs[0], "(a)")

axs[1].set_xlabel(r"$s^{(2)}$")
axs[1].set_ylabel(r"$\frac{\Sigma \cdot T_A }{F^*}$")
plt.sca(axs[1])
plt.legend(title=r"$\Delta$")
make_axes_label(axs[1], "(b)")

plt.savefig("fig6-alt.png", dpi=200)


# In[85]:


fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
cmap = cm.jet
norm = colors.Normalize(0.0, 0.4)

for delta in [0.0, 0.1, 0.2, 0.3, 0.4]:
    file = project.fn(f"prob_rearrang/rdfs-bin-softness-fire_delta-{delta}.pickle")

    data = pickle.load(open(file, "rb"))

    itemps = data["inv_t"]
    in_cuts = data["data"][0]["cuts"]
    f_a = 0.8
    f_b = 0.2
    r = data["r_bc"]
    phi = data["phi"]
    fits = []
    entropy = []
    for i in range(len(data["cuts"])):
        rearrang = [d["rearrang"][i] for d in data["data"]]
        g_a = data["data"][3]["rdfs"][in_cuts[i]]["g_a"]
        g_b = data["data"][3]["rdfs"][in_cuts[i]]["g_b"]
        s = pair_excess_entropy_binary(g_a, g_b, f_a, f_b, r, phi)
        entropy.append(s)
        p = np.polyfit(itemps, np.log(rearrang), 1)

        fits.append(p)

    soft = pickle.load(open(project.fn(f"prob_rearrang/fit_softness_delta-{delta}.pickle"), "rb"))

    def reject_outliers(sr, iq_range=0.8):
        pcnt = (1 - iq_range) / 2
        qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
        iqr = qhigh - qlow
        return sr[ (sr - median).abs() <= iqr]

    data1 = []
    keys = list(soft["data"].keys())
    for i in range(len(keys)-1):
        v1 = soft["data"][keys[i]]["p"]
        for j in range(i+1, len(keys)):
            v2 = soft["data"][keys[j]]["p"]
            data1.append((v2[1] - v1[1]) / (v1[0] - v2[0]))
    T_a = np.mean(reject_outliers(pd.Series(data1)).to_numpy())

    fits = np.array(fits)

    axs[0].plot(entropy, fits[:,1]*(1.2/phi)**(1/3)/(1.0 - delta/2.0**(1/6)), ".", linestyle="", label=f"{delta:.1f}", color=cmap(norm(delta)))
    axs[1].plot(entropy, -fits[:,0]*T_a*(1.2/phi)**(1/3)/(1.0 - delta/2.0**(1/6)), ".", linestyle="", label=f"{delta:.1f}", color=cmap(norm(delta)))

axs[0].set_xlabel(r"$s^{(2)}$")
axs[0].set_ylabel(r"$\frac{\Delta E}{F^*}$")
plt.sca(axs[0])
plt.legend(title=r"$\Delta$")
make_axes_label(axs[0], "(a)")

axs[1].set_xlabel(r"$s^{(2)}$")
axs[1].set_ylabel(r"$\frac{\Sigma \cdot T_A }{F^*}$")
plt.sca(axs[1])
plt.legend(title=r"$\Delta$")
make_axes_label(axs[1], "(b)")

plt.savefig("fig6-alt.png", dpi=200)


# In[30]:


fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=150)
cmap = cm.jet
norm = colors.Normalize(0.0, 0.4)

for delta in [0.0, 0.1, 0.2, 0.3, 0.4]:
    file = project.fn(f"prob_rearrang/rdfs-bin-softness-fire_delta-{delta}.pickle")

    data = pickle.load(open(file, "rb"))

    itemps = data["inv_t"]
    in_cuts = data["data"][0]["cuts"]
    f_a = 0.8
    f_b = 0.2
    r = data["r_bc"]
    phi = data["phi"]
    fits = []
    entropy = []
    for i in range(len(data["cuts"])):
        rearrang = [d["rearrang"][i] for d in data["data"]]
        g_a = data["data"][3]["rdfs"][in_cuts[i]]["g_a"]
        g_b = data["data"][3]["rdfs"][in_cuts[i]]["g_b"]
        s = pair_excess_entropy_binary(g_a, g_b, f_a, f_b, r, phi)
        entropy.append(s)
        p = np.polyfit(itemps, np.log(rearrang), 1)

        fits.append(p)

    soft = pickle.load(open(project.fn(f"prob_rearrang/fit_softness_delta-{delta}.pickle"), "rb"))

    def reject_outliers(sr, iq_range=0.8):
        pcnt = (1 - iq_range) / 2
        qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
        iqr = qhigh - qlow
        return sr[ (sr - median).abs() <= iqr]

    data1 = []
    keys = list(soft["data"].keys())
    for i in range(len(keys)-1):
        v1 = soft["data"][keys[i]]["p"]
        for j in range(i+1, len(keys)):
            v2 = soft["data"][keys[j]]["p"]
            data1.append((v2[1] - v1[1]) / (v1[0] - v2[0]))
    T_a = np.mean(reject_outliers(pd.Series(data1)).to_numpy())

    fits = np.array(fits)

    axs[0].plot(entropy, fits[:,1], ".", linestyle="", label=f"{delta:.1f}", color=cmap(norm(delta)))
    axs[1].plot(entropy, -fits[:,0], ".", linestyle="", label=f"{delta:.1f}", color=cmap(norm(delta)))

axs[1].set_xlabel(r"$s^{(2)}$")
axs[1].set_ylabel(r"$\Delta E$")
plt.sca(axs[0])
# plt.legend(title=r"$\Delta$")
make_axes_label(axs[0], "(a)")

axs[0].set_xlabel(r"$s^{(2)}$")
axs[0].set_ylabel(r"$\Sigma$")
plt.sca(axs[1])
plt.legend(title=r"$\Delta$")
make_axes_label(axs[1], "(b)")

# plt.savefig("fig6-alt2.png", dpi=200)


# # Figure 6 - Diffusion coefficients

# In[ ]:


def plot_reduced_diffusion_coefficient_excess_entropy(ax, diff_data, excess_entropy_folder, legend=True):
    cmap = cm.jet
    norm = mpl.colors.Normalize(vmin=0, vmax=0.4)
    
    for delta, d in diff_data.items():
        phi = delta_to_phi(float(delta))
        with open(excess_entropy_folder + f"/excess-entropy-binary_delta-{delta}.pickle", "rb") as f:
            excess_entropy_data = pickle.load(f)
        plot_data = []
        for temp, data in excess_entropy_data.items():
            plot_data.append(excess_entropy_data[temp]["excess_entropy"])
        ax.plot(np.array(plot_data), np.array(d[()]["D"])*phi**(1/3)*np.sqrt(d[()]["inv_temp"]), label=f"{delta}", color=cmap(norm(float(delta))))
    if legend:
        ax.legend(title=r"$\Delta$")
    ax.set_yscale('log')
    ax.set_xlabel(r"$s^{(2)}$")
    ax.set_ylabel(r"$D^*_A$")

fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=150) # , sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0}
plot_reduced_diffusion_coefficient_softness(axs[0], np.load(project.fn("dynamics/diffusion_coefficients.npz"), allow_pickle=True), project.fn("structure"))
make_axes_label(axs[0], "(a)")
plot_reduced_diffusion_coefficient_excess_entropy(axs[1], np.load(project.fn("dynamics/diffusion_coefficients.npz"), allow_pickle=True), project.fn("structure"))
make_axes_label(axs[1], "(b)")

plt.savefig("fig2.png", dpi=200)


# In[15]:


D


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(8, 3), dpi=150)
cmap = cm.jet
norm = colors.Normalize(0.0, 0.4)

# extract diffusion coefficients
diffusion_coefficients = np.load(project.fn("dynamics/diffusion_coefficients.npz"), allow_pickle=True)

for delta, D_data in diffusion_coefficients.items():
    file = project.fn(f"prob_rearrang/rdfs-bin-softness_delta-{delta}.pickle")

    D = D_data[()]["D"]


    data = pickle.load(open(file, "rb"))

    itemps = data["inv_t"]
    in_cuts = data["data"][0]["cuts"]
    f_a = 0.8
    f_b = 0.2
    r = data["r_bc"]
    phi = data["phi"]
    for j in range(len(itemps)):
        fits = []
        entropy = []
        for i in range(len(data["cuts"])):
            rearrang = [d["rearrang"][i] for d in data["data"]]
            g_a = data["data"][j]["rdfs"][in_cuts[i]]["g_a"]
            g_b = data["data"][j]["rdfs"][in_cuts[i]]["g_b"]
            s = pair_excess_entropy_binary(g_a, g_b, f_a, f_b, r, phi)
            # s = pair_excess_entropy_binary(0.8*g_a + 0.2*g_b, r, phi)
            entropy.append(s)
            p = np.polyfit(itemps, np.log(rearrang), 1)

            fits.append(p)

        soft = pickle.load(open(project.fn(f"prob_rearrang/fit_softness_delta-{delta}.pickle"), "rb"))

        def reject_outliers(sr, iq_range=0.8):
            pcnt = (1 - iq_range) / 2
            qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
            iqr = qhigh - qlow
            return sr[ (sr - median).abs() <= iqr]

        data1 = []
        keys = list(soft["data"].keys())
        for i in range(len(keys)-1):
            v1 = soft["data"][keys[i]]["p"]
            for j in range(i+1, len(keys)):
                v2 = soft["data"][keys[j]]["p"]
                data1.append((v2[1] - v1[1]) / (v1[0] - v2[0]))
        T_a = np.mean(reject_outliers(pd.Series(data1)).to_numpy())

        fits = np.array(fits)

        axs[0].plot(entropy, D[j]*phi**(1/3)*np.sqrt(), ".", linestyle="", label=f"{delta:.1f}", color=cmap(norm(delta)))

axs[0].set_xlabel(r"$s^{(2)}$")
axs[0].set_ylabel(r"$\Delta E$")


# In[38]:


file = project.fn(f"prob_rearrang/rdfs-bin-softness_delta-{delta}.pickle")


# In[39]:


data = pickle.load(open(file, "rb"))


# In[40]:


list(data["data"][0].keys())


# In[43]:


idx = data["data"][0]["cuts"][0]
print(idx)
data["data"][0]['rdfs'][idx].keys()


# In[ ]:





