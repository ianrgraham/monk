import torch
import random
from tqdm import tqdm
from torch import nn, optim
from model import VAE
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from monk import workflow, utils
import pathlib
import signac
import glob
import os
import polars as pl
import numpy as np
import pickle

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
torch.manual_seed(52)

# Hyperparameters
H_DIM = 400
Z_DIM = 1  # without CNN, or some other specialized encoding, 2 is too small. 20 works well for a simple MLP
ALPHA = 0.1

NUM_EPOCHS = 5
BATCH_SIZE = 128
LR = 3e-4  # Karpathy's constant (totally made up)

# collect data
parent = pathlib.Path(os.getcwd()).parent.parent / "config.yaml"
config = workflow.get_config(parent.as_posix())

project: signac.Project = signac.get_project(root=config["root"])

dfs = []

for job in project:
    print(job)
    prep = job.sp["prep"]

    experiments = sorted(glob.glob(job.fn("longer_experiments/*/*/traj-fire_period-*.gsd")))
    if len(experiments) == 0:
        continue

    for exper in experiments:
        max_shear = utils.extract_between(exper, "max-shear-", "/")
        period = utils.extract_between(exper, "period-", ".gsd")
        temp = utils.extract_between(exper, "temp-", "/")

        if (
            float(period) != 1000.0
            or float(temp) != 0.019836
            or float(max_shear) > 0.04
        ):
            continue

        df_paths = glob.glob(job.fn(f"longer_experiments/max-shear-{max_shear}/temp-{temp}/vae-dataset_period-{period}_frame-*.parquet"))

        
        for df_path in df_paths:
            dataset = pl.read_parquet(df_path, use_pyarrow=True)[::10]

            sf_len = dataset[0]["sfs"][0].shape[0]

            X = np.zeros((len(dataset), sf_len + 3 + 1 + 1 + 1), dtype=np.float32)
            X[:, 0] = dataset["strain"]
            X[:, 1] = (dataset["id"] == 0)
            X[:, 2] = (dataset["id"] == 1)
            X[:, 3:-3] = np.vstack(dataset["sfs"].to_numpy())
            X[:, -3] = dataset["frame"]
            X[:, -2] = (prep == "ESL")
            X[:, -1] = dataset["soft"]

            dataset = dataset.with_columns(
                sfs = X,
            )

            dfs.append(dataset)

dataset = pl.concat(dfs)

Y = ((dataset["d2min_left"] > 0.08) | (dataset["d2min_right"] > 0.08)).to_numpy().astype(np.float32)

data_len = len(Y)
sf_len = dataset[0]["sfs"][0].shape[0]

X = np.zeros((data_len, sf_len), dtype=np.float32)
X[:, :] = np.vstack(dataset["sfs"].to_numpy())

# Normalizing Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# scaler_y = StandardScaler()
Y_scaled = Y.reshape(-1, 1)

X_scaled, Y_scaled = shuffle(X_scaled, Y_scaled)

# Converting to PyTorch tensor
X_tensor = torch.FloatTensor(X_scaled)
Y_tensor = torch.FloatTensor(Y_scaled)

import torch.utils.data as data_utils

X_tensor2 = X_tensor.to(device)
Y_tensor2 = Y_tensor.to(device)

torch_dataset = data_utils.TensorDataset(X_tensor2, Y_tensor2)
train_len = int(data_len*0.5)
train, test = data_utils.random_split(torch_dataset, [train_len, data_len - train_len])

train_loader = data_utils.DataLoader(train, batch_size=128, shuffle=True)
test_loader = data_utils.DataLoader(test, batch_size=data_len - train_len)

# Now lets eval!
with open("models/vae-v1-sup.pkl", "rb") as f:
    data = pickle.load(f)

model = VAE(input_dim=data["input_dim"], h_dim=data["h_dim"], z_dim=data["z_dim"]).to(device)
LOAD_PATH = "models/vae-v1.pth"
model.load_state_dict(torch.load(LOAD_PATH))
loss_fn = nn.BCELoss(reduction="mean")  # y_i * log(x_i) + (1 - y_i) * log(1 - x_i)


for X_test, y_test in test_loader:
    y_pred, mu, sigma = model(X_test)
    recon_loss = loss_fn(y_pred, y_test)
    kl_div = - torch.mean(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

    loss = recon_loss + ALPHA * kl_div
    print(f"loss: {loss.item()}, recon_loss: {recon_loss.item()}, kl_div: {kl_div.item()}")