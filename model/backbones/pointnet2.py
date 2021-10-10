### implementation adapted from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_classification.py

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class PointNet2(torch.nn.Module):
    def __init__(self, embed):
        super(PointNet2, self).__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024, embed)
        self.bn1 = nn.BatchNorm1d(embed)

    def forward(self, data):
        data = data.transpose(1, 2)
        B, N, D = data.shape
        pos = data.reshape(B * N, -1)
        batch = torch.arange(B, device=pos.device, dtype=torch.long).repeat_interleave(N)
        x = None

        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        x = F.relu(self.bn1(self.lin1(x)))

        return x