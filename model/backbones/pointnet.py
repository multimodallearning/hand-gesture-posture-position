import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetCls(nn.Module):
    def __init__(self, embed):
        super(PointNetCls, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, embed)
        self.bn4 = nn.BatchNorm1d(embed)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, dim=2, keepdim=False)[0]

        x = F.relu(self.bn4(self.fc1(x)))

        return x
