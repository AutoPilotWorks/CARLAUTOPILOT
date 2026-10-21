from __future__ import annotations

import torch
from torch import nn

from carlautopilot.core import Command

NUM_COMMANDS = len(Command)


class RouteSteeringNet(nn.Module):
    """CIL-style network: shared image+speed features, one output head per route command.

    Output columns: steer in [-1, 1] (positive = right), throttle in [0, 1], brake in [0, 1].
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()
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
