import torch

import torch.nn.functional as F
from torch_geometric.nn.aggr import Aggregation, SumAggregation
from torch_geometric.nn.inits import reset
from torch_geometric.nn import MLP
from torch import nn
import e3nn

class DeepSet(nn.Module):

    def __init__(self, input_dim, output_dim, inter_dim=128, h_dim=32, n_layers=3):
        """Deep set takes input of shape (batch_size, n, input_dim) and returns output of shape (batch_size, output_dim)
        
        The "n" inputs of the set are passed through an MLP and aggregated using sum.

        The sum is then passed through another MLP to get the output.
        """
        super().__init__()

        self.input_dim = input_dim
        self.inter_dim = inter_dim
        self.output_dim = output_dim
        self.h_dim = h_dim
        self.n_layers = n_layers

        self.n_layers = n_layers

        self.mlp1 = MLP(in_channels=self.input_dim, hidden_channels=h_dim,
                        out_channels=self.inter_dim, num_layers=self.n_layers)
        
        self.mlp2 = MLP(in_channels=self.inter_dim, hidden_channels=h_dim,
                        out_channels=self.output_dim, num_layers=self.n_layers)


    def forward(self, x):
        shape = x.shape
        x = self.mlp1(x.view(-1, shape[-1]))
        print(x.shape)
        # sum over middle dimension
        x = torch.sum(x.view(shape[0], shape[1], x.shape[-1]), dim=1)
        print(x.shape)
        x = self.mlp2(x)
        print(x.shape)
        return x



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