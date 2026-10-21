from __future__ import annotations

import numpy as np

HORIZON_FRAC = 0.4
LANE_HALF_WIDTH_M = 1.75
CAMERA_HEIGHT_M = 1.5
FOCAL_FRAC = 1.0


def render_view(center_points: np.ndarray, normals: np.ndarray, h: int = 64, w: int = 128) -> np.ndarray:
    """Pseudo forward camera from route points in the vehicle frame.

    center_points: (N, 2) array of (x forward, y left); normals: (N, 2) unit left normals in the
    vehicle frame. Lane edges are drawn at +-LANE_HALF_WIDTH_M, the centre line dimmer.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    horizon = int(h * HORIZON_FRAC)
    img[:horizon] = (90, 140, 200)
    img[horizon:] = (60, 60, 60)
    focal = FOCAL_FRAC * w
    for offset, color in ((LANE_HALF_WIDTH_M, 255), (-LANE_HALF_WIDTH_M, 255), (0.0, 140)):
        pts = center_points + normals * offset
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        for a, b in zip(pts[:-1], pts[1:]):
            _draw_segment(img, a, b, focal, horizon, h, w, color)
    return img


def _project(p: np.ndarray, focal: float, horizon: int, w: int):
    if p[0] <= 0.5:
        return None
    u = w / 2.0 - focal * p[1] / p[0]
    v = horizon + focal * CAMERA_HEIGHT_M / p[0]
    return u, v


def _draw_segment(img, a, b, focal, horizon, h, w, color) -> None:
    pa, pb = _project(a, focal, horizon, w), _project(b, focal, horizon, w)
    if pa is None or pb is None:
        return
    n = int(max(abs(pa[0] - pb[0]), abs(pa[1] - pb[1]))) + 2
    for t in np.linspace(0.0, 1.0, n):
        u = int(round(pa[0] + (pb[0] - pa[0]) * t))
        v = int(round(pa[1] + (pb[1] - pa[1]) * t))
        if 0 <= u < w and horizon <= v < h:
            img[v, u] = (color, color, color)
