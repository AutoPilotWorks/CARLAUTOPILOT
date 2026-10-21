from __future__ import annotations

import torch
from torch import nn

from carlautopilot.core import Command

NUM_COMMANDS = len(Command)


def _resnet18(feature_dim: int, pretrained: bool) -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18

    net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    net.fc = nn.Sequential(nn.Linear(512, feature_dim), nn.ReLU())
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return nn.Sequential(_Normalize(mean, std), net)


class _Normalize(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class RouteSteeringNet(nn.Module):
    """CIL-style network: shared image+speed features, one output head per route command.

    Output columns: steer in [-1, 1] (positive = right), throttle in [0, 1], brake in [0, 1].
    """

    def __init__(self, feature_dim: int = 128, backbone: str = "small", pretrained: bool = True):
        super().__init__()
        if backbone == "resnet18":
            self.backbone = _resnet18(feature_dim, pretrained)
        elif backbone != "small":
            raise ValueError("backbone must be 'small' or 'resnet18'")
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 4)),
                nn.Flatten(),
                nn.Linear(64 * 8, feature_dim),
                nn.ReLU(),
            )
        self.speed = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        self.merge = nn.Sequential(nn.Linear(feature_dim + 16, feature_dim), nn.ReLU())
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, 3)) for _ in range(NUM_COMMANDS)
        )

    def forward(self, image: torch.Tensor, speed: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        feat = self.merge(torch.cat([self.backbone(image), self.speed(speed.view(-1, 1))], dim=1))
        all_out = torch.stack([head(feat) for head in self.heads], dim=1)
        idx = command.long().view(-1, 1, 1).expand(-1, 1, 3)
        raw = all_out.gather(1, idx).squeeze(1)
        return torch.cat([torch.tanh(raw[:, :1]), torch.sigmoid(raw[:, 1:])], dim=1)
