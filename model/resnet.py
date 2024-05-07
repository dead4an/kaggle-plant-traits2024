# Imports
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Torch options
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: None=3, 
                 stride: None=1, padding: None=1, downsample=None) -> ...:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.downsample = downsample
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out
