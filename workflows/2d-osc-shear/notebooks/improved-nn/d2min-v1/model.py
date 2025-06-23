import torch

import torch.nn.functional as F
from torch import nn


class VAE(nn.Module):

    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2hid1e = nn.Linear(h_dim, h_dim)
        self.hid_2hid2e = nn.Linear(h_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)  # learn mean
        self.hid_2sigma = nn.Linear(h_dim, z_dim)  # learn log variance

        # decoder
        self.z_2hid = nn.Linear(z_dim + 2, h_dim)  # NOTE: +2 for y and soft
        self.hid_2hid1d = nn.Linear(h_dim, h_dim)
        self.hid_2hid2d = nn.Linear(h_dim, h_dim)
        self.hid_2pred = nn.Linear(h_dim, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        h = self.relu(self.hid_2hid1e(h))
        h = self.relu(self.hid_2hid2e(h))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)

        return mu, torch.exp(sigma)

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        h = self.relu(self.hid_2hid1d(h))
        h = self.relu(self.hid_2hid2d(h))
        return self.sigmoid(self.hid_2pred(h))

    def forward(self, x):
        soft = x[:, -1].view(-1, 1)
        y = x[:, 0].view(-1, 1)
        mu, sigma = self.encode(x[:, 1:1 + self.input_dim])
        epsilon = torch.randn_like(sigma)
        z_reparam = mu + sigma * epsilon
        y_pred = self.decode(torch.cat((y, z_reparam, soft), 1))
        return y_pred, mu, sigma
    

class AutoEncoder(nn.Module):
    # input dim is the dim of the feature vector

    def __init__(self, input_dim, add_dim=0, h_dim=200, z_dim=20, act=nn.ReLU(), out_act=nn.Sigmoid()):

        super().__init__()

        self.input_dim = input_dim
        self.add_dim = add_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2hid1e = nn.Linear(h_dim, h_dim)
        self.hid_2hid2e = nn.Linear(h_dim, h_dim)
        self.hid_2z = nn.Linear(h_dim, z_dim)

                # decoder
        self.z_2hid = nn.Linear(z_dim + self.add_dim, h_dim)  # NOTE: +2 for y and soft
        self.hid_2hid1d = nn.Linear(h_dim, h_dim)
        self.hid_2hid2d = nn.Linear(h_dim, h_dim)
        self.hid_2pred = nn.Linear(h_dim, 1)

        self.act = act
        self.out_act = out_act

    def encode(self, x):
        h = self.act(self.img_2hid(x))
        h = self.act(self.hid_2hid1e(h))
        h = self.act(self.hid_2hid2e(h))
        z = self.hid_2z(h)

        return z

    def decode(self, z):
        h = self.act(self.z_2hid(z))
        h = self.act(self.hid_2hid1d(h))
        h = self.act(self.hid_2hid2d(h))
        if self.out_act is None:
            return self.hid_2pred(h)
        return self.out_act(self.hid_2pred(h))

    def forward(self, x):
        y = x[:, :self.add_dim].view(-1, self.add_dim)
        z = self.encode(x[:, self.add_dim:])
        y_pred = self.decode(torch.cat((y, z), 1))
        return y_pred, z
    

class LongAutoEncoder(nn.Module):
    # input dim is the dim of the feature vector

    def __init__(self, input_dim, add_dim=0, h_dim=200, z_dim=20, act=nn.ReLU(), out_act=nn.Sigmoid()):

        super().__init__()

        self.input_dim = input_dim
        self.add_dim = add_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2hid1e = nn.Linear(h_dim, h_dim)
        self.hid_2hid2e = nn.Linear(h_dim, h_dim)
        self.hid_2hid3e = nn.Linear(h_dim, h_dim)
        self.hid_2hid4e = nn.Linear(h_dim, h_dim)
        self.hid_2z = nn.Linear(h_dim, z_dim)

                # decoder
        self.z_2hid = nn.Linear(z_dim + self.add_dim, h_dim)  # NOTE: +2 for y and soft
        self.hid_2hid1d = nn.Linear(h_dim, h_dim)
        self.hid_2hid2d = nn.Linear(h_dim, h_dim)
        self.hid_2hid3d = nn.Linear(h_dim, h_dim)
        self.hid_2hid4d = nn.Linear(h_dim, h_dim)
        self.hid_2pred = nn.Linear(h_dim, 1)

        self.act = act
        self.out_act = out_act

    def encode(self, x):
        h = self.act(self.img_2hid(x))
        h = self.act(self.hid_2hid1e(h))
        h = self.act(self.hid_2hid2e(h))
        h = self.act(self.hid_2hid3e(h))
        h = self.act(self.hid_2hid4e(h))
        z = self.hid_2z(h)

        return z

    def decode(self, z):
        h = self.act(self.z_2hid(z))
        h = self.act(self.hid_2hid1d(h))
        h = self.act(self.hid_2hid2d(h))
        h = self.act(self.hid_2hid3d(h))
        h = self.act(self.hid_2hid4d(h))
        if self.out_act is None:
            return self.hid_2pred(h)
        return self.out_act(self.hid_2pred(h))

    def forward(self, x):
        y = x[:, :self.add_dim].view(-1, self.add_dim)
        z = self.encode(x[:, self.add_dim:])
        y_pred = self.decode(torch.cat((y, z), 1))
        return y_pred, z