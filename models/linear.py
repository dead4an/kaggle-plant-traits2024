# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# Model
class LinearNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(LinearNet, self).__init__()
        self.input_dim = input_dim
        self.layer0 = self._make_layer(input_dim, 356)
        self.layer1 = self._make_layer(356, 512)
        self.layer2 = self._make_layer(512, 356)
        self.layer3 = self._make_layer(356, 128)
        self.layer4 = self._make_layer(128, 128)
        self.layer5 = nn.Linear(128, output_dim)

    def _make_layer(self, input_dim: int, output_dim: int):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        
        return out
