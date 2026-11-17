from __future__ import annotations

import torch
from torch import nn

from carlautopilot.core import Command

NUM_COMMANDS = len(Command)
# perception labels are capped here; a lead/light distance at the cap means 'none in range'
PERCEPT_RANGE_M = 40.0  # a red light 50 m out is often still hidden (crests, small at 160x320); stopping from 29 km/h needs ~20 m
N_PERCEPT = 7  # perception outputs, in episode_store.PERCEPT_KEYS order
N_WAYPOINTS = 8  # path points at fake_env.LOOKAHEAD_OFFSETS_M (3..30 m along the route), x forward / y left m


def _resnet18(feature_dim: int, pretrained: bool) -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18

    net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    net.fc = nn.Sequential(nn.Linear(512, feature_dim), nn.ReLU())
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return nn.Sequential(_Normalize(mean, std), net)


def _backbone(name: str, feature_dim: int, pretrained: bool) -> nn.Module:
    if name == "resnet18":
        return _resnet18(feature_dim, pretrained)
    if name != "small":
        raise ValueError("backbone must be 'small' or 'resnet18'")
    return nn.Sequential(
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


def _light_net(pretrained: bool) -> nn.Module:
    """Traffic-light camera features: ResNet18 up to layer2 (stride 8, batch-normed, ImageNet colour/edge filters), then a
    4x8 grid of per-cell max pools. One max over the whole frame was a constant (sky and bright walls win it, 2026-10-27)."""
    from torchvision.models import ResNet18_Weights, resnet18

    net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return nn.Sequential(_Normalize(mean, std), net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2,
                         nn.Conv2d(128, 32, 1), nn.ReLU(), nn.AdaptiveMaxPool2d((4, 8)), nn.Flatten(),
                         nn.Linear(32 * 32, 64), nn.ReLU())


def _to_unit(x: torch.Tensor | None) -> torch.Tensor | None:
    """uint8 pixels -> float in [0, 1] (same values as the dataset's old CPU conversion); float input passes through.
    The training loader ships uint8 (4x fewer bytes between worker processes) and the GPU converts here. The divisor is
    a tensor because CUDA divides by a Python scalar as x * (1/255), which is 1 ulp off for about half of the 256 values."""
    return x.float() / x.new_tensor(255.0, dtype=torch.float32) if x is not None and x.dtype == torch.uint8 else x


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
    perceive (needs predict_speed) adds 7 perception outputs (PERCEPT_KEYS order): P(red light for my lane within
    PERCEPT_RANGE_M), its stop line m, nearest stop sign m (served or not), P(yield), junction entry m, lead gap m
    (PERCEPT_RANGE_M = none), lead speed relative to ego m/s. They come from a separate head on the image features alone
    (no speed, no command): with the speed input the truck learned 'standing still => stop needed' and never pulled away,
    and one merged 'stop needed' output let it stop at any stop-line look and never restart after a stop sign.
    light_camera (needs perceive) adds the traffic-light image (see _light_net). P(red light) comes from those light
    features alone; the other perception outputs see both images. With P(red) also fed the main image, v1.10 learned
    'stop line / junction ahead => red' (68% right) and ignored the light image completely.
    surround adds the sedan's left/right/rear camera images (B, 3, 3, H, W) in SURROUND_FOLDERS order: one encoder shared
    by the three (its own weights, not the front backbone's: they look down and sideways), features concatenated in that
    fixed order next to the front features, so the driving heads and the perception head (except P(red)) see all of them.
    waypoints adds the navigation target point (route 30 m ahead, vehicle frame, B x 2) as an input and N_WAYPOINTS path
    points as the last 2 * N_WAYPOINTS output columns (x0, y0, x1, ...), one head per command like the driving heads; a
    controller then follows the points (LBC/TCP-style). Steering regressed directly never saw a way back to the lane and
    turned late into junctions: 22 of v2.1 A's 28 failures began with leaving the lane (11-17).
    """

    def __init__(self, feature_dim: int = 128, backbone: str = "small", pretrained: bool = True, predict_speed: bool = False,
                 perceive: bool = False, light_camera: bool = False, surround: bool = False, waypoints: bool = False):
        super().__init__()
        if perceive and not predict_speed:
            raise ValueError("perceive needs predict_speed")
        if light_camera and not perceive:
            raise ValueError("light_camera needs perceive")
        self.predict_speed = predict_speed
        self.perceive = perceive
        self.light_camera = light_camera
        self.waypoints = waypoints
        self.n_out = (4 + N_PERCEPT if perceive else 4 if predict_speed else 3) + (2 * N_WAYPOINTS if waypoints else 0)
        head_out = 4 if predict_speed else 3
        self.backbone = _backbone(backbone, feature_dim, pretrained)
        self.surround = surround
        if surround:
            self.surround_net = _backbone(backbone, feature_dim, pretrained)
        img_dim = feature_dim * (4 if surround else 1)
        self.speed = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        if waypoints:
            self.target = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
            self.wp_heads = nn.ModuleList(
                nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, 2 * N_WAYPOINTS)) for _ in range(NUM_COMMANDS)
            )
        self.merge = nn.Sequential(nn.Linear(img_dim + 16 + (16 if waypoints else 0), feature_dim), nn.ReLU())
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, head_out)) for _ in range(NUM_COMMANDS)
        )
        if perceive:
            light_dim = 64 if light_camera else 0
            self.percept = nn.Sequential(nn.Linear(img_dim + light_dim, 64), nn.ReLU(), nn.Linear(64, N_PERCEPT))
        if light_camera:
            self.light_net = _light_net(pretrained)
            self.light_red = nn.Linear(64, 1)

    def forward(self, image: torch.Tensor, speed: torch.Tensor, command: torch.Tensor, light: torch.Tensor | None = None,
                surround: torch.Tensor | None = None, target_point: torch.Tensor | None = None) -> torch.Tensor:
        image, light, surround = _to_unit(image), _to_unit(light), _to_unit(surround)
        img = self.backbone(image)
        if self.surround:
            if surround is None:
                raise ValueError("this model needs the surround camera images")
            img = torch.cat([img, self.surround_net(surround.flatten(0, 1)).view(len(img), -1)], dim=1)
        parts = [img, self.speed(speed.view(-1, 1))]
        if self.waypoints:
            if target_point is None:
                raise ValueError("this model needs the navigation target point")
            parts.append(self.target(target_point.view(-1, 2) / 30.0))  # metres, ~unit scale
        feat = self.merge(torch.cat(parts, dim=1))

        def by_command(heads):
            out = torch.stack([head(feat) for head in heads], dim=1)
            return out.gather(1, command.long().view(-1, 1, 1).expand(-1, 1, out.shape[2])).squeeze(1)

        raw = by_command(self.heads)
        if self.perceive:
            if self.light_camera:
                if light is None:
                    raise ValueError("this model needs the traffic-light camera image")
                lf = self.light_net(light)
                p = self.percept(torch.cat([img, lf], dim=1))
                p = torch.cat([self.light_red(lf), p[:, 1:]], dim=1)  # column 0 is P(red)
            else:
                p = self.percept(img)
            raw = torch.cat([raw, p], dim=1)
        out = [torch.tanh(raw[:, :1]), torch.sigmoid(raw[:, 1:3])]
        if self.predict_speed:
            out.append(nn.functional.softplus(raw[:, 3:4]))
        if self.perceive:
            out += [torch.sigmoid(raw[:, 4:5]), nn.functional.softplus(raw[:, 5:7]), torch.sigmoid(raw[:, 7:8]),
                    nn.functional.softplus(raw[:, 8:10]), raw[:, 10:11]]
        if self.waypoints:
            out.append(by_command(self.wp_heads))
        return torch.cat(out, dim=1)
