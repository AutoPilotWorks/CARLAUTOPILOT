from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from carlautopilot.core import Action, Command, Observation
from carlautopilot.data.episode_store import SURROUND_FOLDERS

TILE = (320, 160)  # w, h of each camera tile; the light crop is resized to it
YELLOW, WHITE, GREY, RED, GREEN, BLUE = (255, 220, 0), (240, 240, 240), (150, 150, 150), (255, 70, 60), (60, 210, 90), (0, 190, 255)


def compose(obs: Observation, action: Action, target_speed: float | None = None, percept: tuple[float, ...] | None = None) -> Image.Image:
    """One frame: LEFT FRONT RIGHT over LIGHT REAR and a panel with what the policy decided.
    (L) = learned by the network, (R) = rule (ACC pedals, stop-sign serving)."""
    blank = np.zeros((TILE[1], TILE[0], 3), np.uint8)
    views = obs.extras.get("surround_images") or {}
    tile = lambda img: Image.fromarray(blank if img is None else np.asarray(img, np.uint8)).resize(TILE)
    tiles = [("LEFT", views.get(SURROUND_FOLDERS[0])), ("FRONT", obs.image), ("RIGHT", views.get(SURROUND_FOLDERS[1])),
             ("LIGHT", obs.extras.get("light_image")), ("REAR", views.get(SURROUND_FOLDERS[2]))]
    canvas = Image.new("RGB", (TILE[0] * 3, TILE[1] * 2), (20, 20, 20))
    d = ImageDraw.Draw(canvas)
    font = _font(14)
    for k, (name, img) in enumerate(tiles):
        x, y = TILE[0] * (k % 3), TILE[1] * (k // 3)
        canvas.paste(tile(img), (x, y))
        d.text((x + 6, y + 4), name, fill=YELLOW, font=font)
    x0, y = TILE[0] * 2 + 10, TILE[1] + 6
    d.text((x0, y), f"{obs.speed_mps * 3.6:4.0f} km/h   {Command(int(obs.command)).name}", fill=WHITE, font=_font(18))
    y += 26
    y = _bar(d, x0, y, "steer (L)", action.steer, -1.0, BLUE, font)
    y = _bar(d, x0, y, "throttle (R)", action.throttle, 0.0, GREEN, font)
    y = _bar(d, x0, y, "brake (R)", action.brake, 0.0, RED, font)
    if target_speed is not None:
        d.text((x0, y), f"target speed {target_speed * 3.6:3.0f} km/h (L)", fill=WHITE, font=font)
        y += 17
    if percept is not None and len(percept) == 7:
        p_red, red_d, sign_d, p_yield, yield_d, lead_d, lead_rel = percept
        d.text((x0, y), f"red {p_red:4.0%}" + (f" @ {red_d:3.0f} m" if p_red > 0.5 else ""), fill=RED if p_red > 0.5 else GREY, font=font)
        d.text((x0 + 150, y), f"yield {p_yield:4.0%}" + (f" @ {yield_d:3.0f} m" if p_yield > 0.5 else ""), fill=YELLOW if p_yield > 0.5 else GREY, font=font)
        y += 17
        sign = f"stop sign {sign_d:3.0f} m" if sign_d < 38.0 else "stop sign -"
        lead = f"lead {lead_d:3.0f} m {lead_rel * 3.6:+4.0f} km/h" if lead_d < 38.0 else "lead -"
        d.text((x0, y), sign, fill=WHITE if sign_d < 38.0 else GREY, font=font)
        d.text((x0 + 150, y), lead, fill=WHITE if lead_d < 38.0 else GREY, font=font)
    return canvas


def _bar(d: ImageDraw.ImageDraw, x: int, y: int, label: str, v: float, lo: float, colour, font) -> int:
    d.text((x, y), label, fill=WHITE, font=font)
    bx, w = x + 100, 190
    d.rectangle((bx, y + 3, bx + w, y + 13), outline=GREY)
    zero = bx + (w // 2 if lo < 0 else 0)
    end = zero + int(v * (w // 2 if lo < 0 else w))
    d.rectangle((min(zero, end), y + 4, max(zero, end), y + 12), fill=colour)
    return y + 19


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


class LiveView:
    """A window that shows each step as the policy saw and decided it. Display only: closing it does not stop the run.
    It costs some CPU per step; results do not change because CARLA runs synchronously."""

    def __init__(self, title: str = "carlautopilot", scale: float = 1.5, every: int = 2):
        """every: draw one step in this many (2 = 10 frames/s at dt 0.05); a frame costs ~28 ms of CPU (2026-10-31)."""
        import tkinter as tk

        from PIL import ImageTk

        self._tk, self._imagetk = tk, ImageTk
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root, bg="black")
        self.label.pack()
        self.scale, self.every, self.closed, self._n = scale, every, False, 0
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def show(self, obs: Observation, action: Action, policy=None) -> None:
        self._n += 1
        if self.closed or obs.image is None or (self._n - 1) % self.every:
            return
        img = compose(obs, action, getattr(policy, "last_target_speed", None), getattr(policy, "last_percept", None))
        img = img.resize((int(img.width * self.scale), int(img.height * self.scale)), Image.BILINEAR)
        try:
            self._photo = self._imagetk.PhotoImage(img)  # keep a reference or Tk drops the image
            self.label.configure(image=self._photo)
            self.root.update()
        except self._tk.TclError:
            self.closed = True

    def close(self) -> None:
        self.closed = True
        try:
            self.root.destroy()
        except self._tk.TclError:
            pass
