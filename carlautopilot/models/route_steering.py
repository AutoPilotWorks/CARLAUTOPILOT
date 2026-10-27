from __future__ import annotations

import torch
from torch import nn

from carlautopilot.core import Command

NUM_COMMANDS = len(Command)
# perception labels are capped here; a lead/light distance at the cap means 'none in range'
PERCEPT_RANGE_M = 40.0  # a red light 50 m out is often still hidden (crests, small at 160x320); stopping from 29 km/h needs ~20 m


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

    Output columns: steer in [-1, 1] (positive = right), throttle in [0, 1], brake in [0, 1],
    and with predict_speed a 4th: the road-geometry target speed in m/s (>= 0).
    perceive (needs predict_speed) adds 4 more: P(red light for my lane within PERCEPT_RANGE_M), distance to its
    stop line m, gap to the lead vehicle/pedestrian m (PERCEPT_RANGE_M = none), lead speed relative to ego m/s.
    percept_from_image computes those 4 from the image features alone (no speed, no command): with the speed input the
    model learned 'standing still => stop needed' (stopped frames are mostly at red lights) and never pulled away.
    split_stops (needs percept_from_image) replaces those 4 with 7 by cause: P(red light), its stop line m, nearest stop
    sign m (served or not; PERCEPT_RANGE_M = none), P(yield), junction entry m, lead gap m, lead relative speed m/s.
    One merged 'stop needed' let the model stop at any stop line look and never restart after a stop sign.
    light_camera (needs percept_from_image) adds a second, narrow image for traffic lights: a small full-resolution CNN
    with max pooling (a light head is a few pixels; average pooling would wash it out) feeds only the perception head.
    """

    def __init__(self, feature_dim: int = 128, backbone: str = "small", pretrained: bool = True, predict_speed: bool = False,
                 perceive: bool = False, percept_from_image: bool = False, split_stops: bool = False,
                 light_camera: bool = False):
        super().__init__()
        if perceive and not predict_speed:
            raise ValueError("perceive needs predict_speed")
        if percept_from_image and not perceive:
            raise ValueError("percept_from_image needs perceive")
        if split_stops and not percept_from_image:
            raise ValueError("split_stops needs percept_from_image")
        self.predict_speed = predict_speed
        self.perceive = perceive
        self.percept_from_image = percept_from_image
        self.split_stops = split_stops
        if light_camera and not percept_from_image:
            raise ValueError("light_camera needs percept_from_image")
        self.light_camera = light_camera
        n_percept = 7 if split_stops else 4
        self.n_out = 4 + n_percept if perceive else 4 if predict_speed else 3
        head_out = 4 if percept_from_image else self.n_out
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
            nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, head_out)) for _ in range(NUM_COMMANDS)
        )
        if percept_from_image:
            light_dim = 64 if light_camera else 0
            self.percept = nn.Sequential(nn.Linear(feature_dim + light_dim, 64), nn.ReLU(), nn.Linear(64, n_percept))
        if light_camera:
            self.light_net = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveMaxPool2d((2, 4)), nn.Flatten(), nn.Linear(64 * 8, 64), nn.ReLU(),
            )

    def forward(self, image: torch.Tensor, speed: torch.Tensor, command: torch.Tensor, light: torch.Tensor | None = None) -> torch.Tensor:
        img = self.backbone(image)
        feat = self.merge(torch.cat([img, self.speed(speed.view(-1, 1))], dim=1))
        all_out = torch.stack([head(feat) for head in self.heads], dim=1)
        idx = command.long().view(-1, 1, 1).expand(-1, 1, all_out.shape[2])
        raw = all_out.gather(1, idx).squeeze(1)
        if self.percept_from_image:
            if self.light_camera:
                if light is None:
                    raise ValueError("this model needs the traffic-light camera image")
                pin = torch.cat([img, self.light_net(light)], dim=1)
            else:
                pin = img
            raw = torch.cat([raw, self.percept(pin)], dim=1)
        out = [torch.tanh(raw[:, :1]), torch.sigmoid(raw[:, 1:3])]
        if self.predict_speed:
            out.append(nn.functional.softplus(raw[:, 3:4]))
        if self.split_stops:
            out += [torch.sigmoid(raw[:, 4:5]), nn.functional.softplus(raw[:, 5:7]), torch.sigmoid(raw[:, 7:8]),
                    nn.functional.softplus(raw[:, 8:10]), raw[:, 10:11]]
        elif self.perceive:
            out += [torch.sigmoid(raw[:, 4:5]), nn.functional.softplus(raw[:, 5:7]), raw[:, 7:8]]
        return torch.cat(out, dim=1)
