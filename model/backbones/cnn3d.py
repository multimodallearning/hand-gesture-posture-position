import torch
import torch.nn as nn


class VoxNet(nn.Module):
    def __init__(self, embed):
        super(VoxNet, self).__init__()
        self.features = nn.ModuleList([
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(2)
        ])
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.ModuleList([
            nn.Linear(in_features=512, out_features=embed),
            nn.BatchNorm1d(embed),
            nn.ReLU()
        ])

    def forward(self, x):
        # voxelize point cloud into a 32x32x32 grid; the input cloud is expected to be normalized to [-1, 1]
        B = x.size(0)
        x = x.transpose(1,2)
        x -= torch.tensor([[[-1., -1., -1.]]], device=x.device, dtype=x.dtype)
        x = x / (2 / 32)
        x = torch.floor(x).long()
        x = torch.clamp(x, 0, 31)
        vol = torch.zeros([B, 1, 32, 32, 32], device=x.device)
        for i in range(B):
            vol[i, 0, x[i, :, 0], x[i, :, 1], x[i, :, 2]] = 1.
        vol = vol.float()

        for layer in self.features:
            vol = layer(vol)
        vol = self.pool(vol)
        vol = torch.flatten(vol, 1)
        for layer in self.fc:
            vol = layer(vol)

        return vol
