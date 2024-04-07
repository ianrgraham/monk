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

