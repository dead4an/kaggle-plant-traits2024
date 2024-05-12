import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiInputResNet(nn.Module):
    def __init__(self, backbone: nn.Module, linear: nn.Module, 
                 linear_output_dim: int, output_dim: int) -> None:
        super(MultiInputResNet, self).__init__()
        self.backbone = backbone
        self.linear = linear
        self.fc0 = self._make_linear_layer(1000 + linear_output_dim, 1024)
        self.fc1 = self._make_linear_layer(1024, 512)
        self.fc2 = self._make_linear_layer(512, 256)
        self.fc3 = self._make_linear_layer(256, 128)
        self.fc4 = self._make_linear_layer(128, 64)
        self.fc5 = nn.Linear(64, output_dim)

    def forward(self, img, features) -> torch.Tensor:
        backbone_out = self.backbone(img)
        linear_out = self.linear(features)
        out = torch.cat((backbone_out, linear_out), dim=-1)

        out = self.fc0(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)

        return out
    
    def _make_linear_layer(self, input_dim: int, 
                           output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
