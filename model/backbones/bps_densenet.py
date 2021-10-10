import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bps import bps


def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


class BPSDensenet(nn.Module):

    def __init__(self, cfg, embed):
        super(BPSDensenet, self).__init__()
        num_bps_points = cfg.MODEL.BPS_DENSENET.NUM_BPS_POINTS
        self.basis = bps.generate_random_basis(num_bps_points, n_dims=3,radius=1.,
                                               random_seed=cfg.MODEL.BPS_DENSENET.RANDOM_SEED)
        self.basis = torch.from_numpy(np.float32(self.basis)).to(cfg.MODEL.DEVICE)

        hsize = 256
        self.num_bps_stages = 2
        self.num_layers_per_stage = 2

        self.bn0 = nn.BatchNorm1d(num_bps_points)
        self.densenet = nn.ModuleList([])
        in_f = num_bps_points
        for i in range(self.num_bps_stages):
            self.densenet.append(nn.Linear(in_features=in_f, out_features=hsize))
            in_f += hsize
            self.densenet.append(nn.BatchNorm1d(hsize))
            for _ in range(self.num_layers_per_stage - 1):
                self.densenet.append(nn.Linear(in_features=hsize, out_features=hsize))
                self.densenet.append(nn.BatchNorm1d(hsize))
        self.densenet.append(nn.Linear(in_features=in_f, out_features=embed))
        self.densenet.append(nn.BatchNorm1d(embed))

    def forward(self, x):
        # encode input cloud with basis point set
        x = x.transpose(1,2)
        B, N, D = x.size()
        x = x.view(-1, D)
        dist_matrix = pairwise_distances(x, self.basis)
        dist_matrix = dist_matrix.view(B, N, self.basis.size(0))
        x = torch.sqrt(torch.min(dist_matrix, dim=1)[0])

        # feature extraction with densenet
        x = self.bn0(x)
        if self.densenet is not None:
            x_0 = [x]
            for i in range(self.num_bps_stages):
                x = torch.cat(x_0, dim=1)
                for j in range(self.num_layers_per_stage):
                    fc_idx = 2 * i * self.num_layers_per_stage + 2 * j
                    fc = self.densenet[fc_idx]
                    bn = self.densenet[fc_idx + 1]
                    x = bn(F.relu(fc(x)))
                x_0.append(x)
            x = torch.cat(x_0, dim=1)
            x = self.densenet[-2](x)
            x = F.relu(x)
            x = self.densenet[-1](x)

        x = x.view(B, -1)
        return x
