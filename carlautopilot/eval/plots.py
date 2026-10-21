from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from carlautopilot.eval.episode_log import read_episode

W, H = 900, 760
BG, FG, GRID = (255, 255, 255), (30, 30, 30), (220, 220, 220)
COLORS = {"speed": (30, 100, 200), "lat": (200, 80, 30), "path": (40, 40, 40)}
EVENT_COLOR = (210, 30, 30)


def _scale(vals, lo_px, hi_px, pad=0.0):
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-6)
    lo -= span * pad
    span *= 1 + 2 * pad
    return lambda v: lo_px + (v - lo) / span * (hi_px - lo_px)


def _series(d, box, ts, ys, color, title, ymin=None, ymax=None):
    x0, y0, x1, y1 = box
    d.rectangle(box, outline=GRID)
    d.text((x0 + 4, y0 + 2), title, fill=FG)
    lo = min(ys) if ymin is None else ymin
    hi = max(ys) if ymax is None else ymax
    if hi - lo < 1e-6:
        hi = lo + 1.0
    sx = _scale([ts[0], ts[-1] if ts[-1] > ts[0] else ts[0] + 1.0], x0 + 4, x1 - 4)
    sy = lambda v: y1 - 4 - (v - lo) / (hi - lo) * (y1 - y0 - 24)
    d.text((x0 + 4, y1 - 14), f"{ts[0]:.0f}s", fill=FG)
    d.text((x1 - 40, y1 - 14), f"{ts[-1]:.0f}s", fill=FG)
    d.text((x1 - 90, y0 + 2), f"{lo:.1f}..{hi:.1f}", fill=FG)
    d.line([(sx(t), sy(v)) for t, v in zip(ts, ys)], fill=color, width=2)
    return sx


def plot_log(log_path: str | Path, out_png: str | Path) -> Path:
    header, recs = read_episode(log_path)
    if not recs:
        raise ValueError("log has no steps")
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    xs, ys = [r.truth.x for r in recs], [r.truth.y for r in recs]
    box = (10, 24, W - 10, 440)
    d.text((10, 6), f"{header.scenario_name}  policy={header.policy_name}  steps={len(recs)}", fill=FG)
    d.rectangle(box, outline=GRID)
    # equal axis scale so curves keep their shape
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    k = min((box[2] - box[0] - 20) / span, (box[3] - box[1] - 20) / span)
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    mx, my = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    pt = lambda x, y: (mx + (x - cx) * k, my - (y - cy) * k)
    d.line([pt(x, y) for x, y in zip(xs, ys)], fill=COLORS["path"], width=2)
    d.ellipse([pt(xs[0], ys[0])[0] - 4, pt(xs[0], ys[0])[1] - 4, pt(xs[0], ys[0])[0] + 4, pt(xs[0], ys[0])[1] + 4], fill=(30, 160, 60))
    for r in recs:
        for e in r.events:
            px, py = pt(r.truth.x, r.truth.y)
            d.ellipse([px - 4, py - 4, px + 4, py + 4], outline=EVENT_COLOR, width=2)
            d.text((px + 6, py - 6), e.kind, fill=EVENT_COLOR)
    ts = [r.t for r in recs]
    _series(d, (10, 450, W - 10, 590), ts, [r.speed_mps for r in recs], COLORS["speed"], "speed [m/s]", ymin=0.0)
    sx = _series(d, (10, 600, W - 10, 740), ts, [r.truth.lateral_error_m for r in recs], COLORS["lat"], "lateral error [m]")
    for r in recs:
        for e in r.events:
            for top, bottom in ((450, 590), (600, 740)):
                d.line([(sx(r.t), top), (sx(r.t), bottom)], fill=EVENT_COLOR)
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def plot_logs(log_dir: str | Path, out_dir: str | Path) -> list[Path]:
    return [plot_log(p, Path(out_dir) / f"{p.stem}.png") for p in sorted(Path(log_dir).glob("*.jsonl"))]
