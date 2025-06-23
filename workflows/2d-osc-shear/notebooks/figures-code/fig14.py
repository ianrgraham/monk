# bring in imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from scipy.stats import pearsonr
from scipy import stats

import glob

from dataclasses import dataclass
from collections import defaultdict
import pathlib
import os
import signac

from monk import utils, workflow
from tqdm import tqdm
import pickle

qle_files = glob.glob("/media/ian/Poppy/improved-nn/qle-v2/*/*.npz")

keep_files = []
for file in qle_files:
    data = np.load(file)
    if np.mean(data["f16"]) < 1.0:
        keep_files.append(file)

qle_files = keep_files

d2min_files = [file.replace("qle", "d2min").replace("npz", "parquet").replace("-v2", "-v1") for file in qle_files]

@dataclass(frozen=True, eq=True)
class Statepoint:
    max_shear: float
    period: float
    temp: float
    prep: str
    cycle: int

# grab project

cwd = pathlib.Path(os.getcwd())
parent = cwd.parent.parent / "config.yaml"
config = workflow.get_config(parent.as_posix())

project: signac.Project = signac.get_project(root=config['root'])

prep_dict = {}
for job in project:
    prep = job.sp["prep"]
    if prep == "ESL":
        prep = "GQ"
    prep_dict[job.id] = prep

cache_output = glob.glob(".data/fig4-cache.pkl")
if len(cache_output) > 0:
    with open(cache_output[0], "rb") as f:
        output = pickle.load(f)
else:
    output = defaultdict(list)
    for idx in tqdm(range(len(qle_files))):
        exper = qle_files[idx]
        qle_data = np.load(exper)
        d2min_data = pd.read_parquet(d2min_files[idx])
        max_shear = utils.extract_between(exper, "max-shear-", "_")
        period = utils.extract_between(exper, "period-", "_")
        temp = utils.extract_between(exper, "temp-", "_")
        cycle = (int(utils.extract_between(exper, "frame-", ".")) + 1 ) // 40
        prep = prep_dict[exper.split("/")[-2]]
        sp = Statepoint(max_shear=float(max_shear), period=float(period), temp=float(temp), prep=prep, cycle=cycle)
        if float(temp) > 1e-2:
            continue
        new_df = d2min_data[["soft", "d2min_irr", "d2min_rev", "d2min_left", "d2min_right"]]
        new_df["f512"] = qle_data["f256"]
        output[sp].append(new_df)
        # break

    with open(".data/fig4-cache.pkl", "wb") as f:
        pickle.dump(output, f)

fig, all_axs = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)

axs = all_axs[0]

cmap = cm.magma
norm = colors.Normalize(vmin=0, vmax=20)

#remove padding be

d2min_name = "d2min_irr"

@dataclass(frozen=True, eq=True)
class Plot:
    prep: str
    cycle: int

plot_out = defaultdict(list)

from sklearn.linear_model import LogisticRegression

for key, val in output.items():

    df = pd.concat(val)

    x = np.log(df["f512"]).values.reshape(-1, 1)
    x2 = df["soft"].values.reshape(-1, 1)
    x = np.concatenate([x, x2], axis=1)
    y = df[d2min_name] > 0.08
    model = LogisticRegression(class_weight="balanced").fit(x, y)

    # pr = stats.spearmanr(, np.log(df[d2min_name]))
    plot_out[Plot(prep=key.prep, cycle=key.cycle)].append((key.max_shear, model.score(x, y)))
    # plt.plot(key.max_shear, pr[0], marker=m, color=c)

plt.sca(axs[0])
plt.ylabel(r"$S_{D^2_{min;strob}}$", size="xx-large")
# plt.xlabel(r"$\gamma_{max}$", size="xx-large")
plt.title(r"$\mathcal{M}$", size="xx-large")

for key, val in plot_out.items():
    if key.prep == "GQ":
        ls = "-"
        
    else:
        m = "."
        ls = ":"
    # ls = "-"

    c = cmap(norm(key.cycle))
    if key.cycle in [5, 13]:
        continue

    val = sorted(val, key=lambda x: x[0])
    x, y = zip(*val)
    plt.plot(x, y, color=c, ls=ls, marker=".")

plot_out = defaultdict(list)

for key, val in output.items():

    df = pd.concat(val)
    x = df["soft"].values.reshape(-1, 1)
    y = df[d2min_name] > 0.08
    model = LogisticRegression(class_weight="balanced").fit(x, y)
    
    # pr = stats.spearmanr(, np.log(df[d2min_name]))
    plot_out[Plot(prep=key.prep, cycle=key.cycle)].append((key.max_shear, model.score(x, y)))
    # plt.plot(key.max_shear, pr[0], marker=m, color=c)

plt.sca(axs[1])
plt.title(r"$S$", size="xx-large")

for key, val in plot_out.items():
    if key.prep == "GQ":
        prep = "GQ"
        ls = "-"
        # ls = "-"
    else:
        prep = "HTL"
        ls = ":"
    # ls = "-"

    # if key.cycle == 1:
    #     c = "green"
    # else:
    #     c = "purple"
    c = cmap(norm(key.cycle))
    if key.cycle in [5, 13]:
        continue

    val = sorted(val, key=lambda x: x[0])
    x, y = zip(*val)
    plt.plot(x, y, color=c, marker=".", ls=ls, label=f"{prep} {key.cycle-1}")

# ax1 = plt.gca()

# plt.legend(title="prep cycle")
# plt.axis('on')
# ax1.set_xticklabels([])
# ax1.set_yticklabels([])
# ax1.set_aspect('equal')
handles, labels = plt.gca().get_legend_handles_labels()
# resort 
# order = [3, 0, 2, 1, 4, 8, 5, 7, 6, 9]
order, _ = zip(*sorted(enumerate(labels), key=lambda x: x[1]))
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], title="(PREP, CYCLE)", loc="upper left", bbox_to_anchor=(1.05, 1))
plt.subplots_adjust(wspace=0, hspace=0)

# plt.xlabel(r"$\gamma_{max}$", size="xx-large")

# plt.savefig("corr-irr-256.png", dpi=200, bbox_inches="tight")

# fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
axs = all_axs[1]

cmap = cm.magma
norm = colors.Normalize(vmin=0, vmax=20)
#remove padding be

d2min_name = "d2min_rev"

plot_out = defaultdict(list)

for key, val in output.items():

    df = pd.concat(val)
    x = np.log(df["f512"]).values.reshape(-1, 1)
    x2 = df["soft"].values.reshape(-1, 1)
    x = np.concatenate([x, x2], axis=1)
    y = df[d2min_name] > 0.08
    model = LogisticRegression(class_weight="balanced").fit(x, y)
    
    # pr = stats.spearmanr(, np.log(df[d2min_name]))
    plot_out[Plot(prep=key.prep, cycle=key.cycle)].append((key.max_shear, model.score(x, y)))
    # plt.plot(key.max_shear, pr[0], marker=m, color=c)

plt.sca(axs[0])
plt.ylabel(r"$S_{D^2_{min;ptp}}$", size="xx-large")
plt.xlabel(r"$\gamma_{max}$", size="xx-large")
# plt.title(r"$\mathcal{M}$", size="xx-large")

for key, val in plot_out.items():
    if key.prep == "GQ":
        ls = "-"
        
    else:
        ls = ":"
        # ls = ":"
    # ls = "-"

    # if key.cycle == 1:
    #     c = "green"
    # else:
    #     c = "purple"
    c = cmap(norm(key.cycle))
    if key.cycle in [5, 13]:
        continue

    val = sorted(val, key=lambda x: x[0])
    x, y = zip(*val)
    plt.plot(x, y, color=c, ls=ls, marker=".")

plot_out = defaultdict(list)

for key, val in output.items():

    df = pd.concat(val)

    x = df["soft"].values.reshape(-1, 1)
    y = df[d2min_name] > 0.08
    model = LogisticRegression(class_weight="balanced").fit(x, y)
    
    # pr = stats.spearmanr(, np.log(df[d2min_name]))
    plot_out[Plot(prep=key.prep, cycle=key.cycle)].append((key.max_shear, model.score(x, y)))
    # plt.plot(key.max_shear, pr[0], marker=m, color=c)

plt.sca(axs[1])
# plt.title(r"$S$", size="xx-large")

for key, val in plot_out.items():
    if key.prep == "GQ":
        prep = "GQ"
        ls = "-"
        # ls = "-"
    else:
        prep = "HTL"
        ls = ":"
    # ls = "-"

    # if key.cycle == 1:
    #     c = "green"
    # else:
    #     c = "purple"
    c = cmap(norm(key.cycle))
    if key.cycle in [5, 13]:
        continue

    val = sorted(val, key=lambda x: x[0])
    x, y = zip(*val)
    plt.plot(x, y, color=c, ls=ls, marker=".", label=f"{prep} {key.cycle-1}")

# ax1 = plt.gca()

# plt.legend(title="(PREP, CYCLE)", loc="upper left", bbox_to_anchor=(1.05, 1))
# plt.axis('on')
# ax1.set_xticklabels([])
# ax1.set_yticklabels([])
# ax1.set_aspect('equal')
# plt.subplots_adjust(wspace=0)

plt.xlabel(r"$\gamma_{max}$", size="xx-large")

plt.savefig("FIG15v3.png", dpi=200, bbox_inches="tight")