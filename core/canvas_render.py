"""Rasterizes vector strokes with OpenCV and exports PNGs (no UI toolkit).

Layers are *premultiplied* RGBA: drawing a color with anti-aliasing onto a
transparent (0, 0, 0, 0) layer scales the color and alpha together. That makes
compositing a single expression and lets the eraser simply paint (0, 0, 0, 0).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image

from core.canvas import DrawingCanvas, Stroke

RGB = tuple[int, int, int]


def stroke_width_px(stroke: Stroke, height: int) -> int:
    return max(1, int(round(stroke.thickness * height)))


def draw_stroke(layer: np.ndarray, stroke: Stroke) -> None:
    """Draw one stroke onto a premultiplied RGBA layer in place."""
    h, w = layer.shape[:2]
    color = (0, 0, 0, 0) if stroke.erase else (*stroke.color, 255)
    width = stroke_width_px(stroke, h)
    pts = np.array([(x * (w - 1), y * (h - 1)) for x, y in stroke.points], dtype=np.float64)
    pts_i = np.round(pts).astype(np.int32)
    if len(pts_i) == 1:
        cv2.circle(layer, tuple(int(v) for v in pts_i[0]), max(1, width // 2), color, -1, cv2.LINE_AA)
    else:
        cv2.polylines(layer, [pts_i.reshape(-1, 1, 2)], False, color, width, cv2.LINE_AA)


def render_strokes(strokes: Iterable[Stroke], width: int, height: int) -> np.ndarray:
    layer = np.zeros((height, width, 4), dtype=np.uint8)
    for stroke in strokes:
        draw_stroke(layer, stroke)
    return layer


def composite(frame_rgb: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Blend a premultiplied RGBA layer over an RGB frame of the same size.

    out = frame * (1 - alpha) + layer_rgb, computed with OpenCV's saturating
    SIMD arithmetic (several times faster than NumPy for a full frame).
    """
    alpha = cv2.extractChannel(layer, 3)
    if cv2.countNonZero(alpha) == 0:
        return frame_rgb
    inverse = cv2.merge([cv2.bitwise_not(alpha)] * 3)  # 255 - alpha
    background = cv2.multiply(frame_rgb, inverse, scale=1.0 / 255.0)
    return cv2.add(background, cv2.cvtColor(layer, cv2.COLOR_RGBA2RGB))


class LayerCache:
    """Caches the rasterized drawing for the current display size.

    * Committed strokes are re-rendered only when they or the size change.
    * The stroke being drawn is rendered *incrementally*: each frame only the
      segments added since the previous frame are drawn. Before, the whole
      active stroke was redrawn every frame, so the cost grew with the length
      of a continuous line (up to 4000 points).
    """

    def __init__(self) -> None:
        self._key: Optional[tuple[int, int, int]] = None
        self._layer: Optional[np.ndarray] = None
        self._active_layer: Optional[np.ndarray] = None
        self._active_signature: Optional[tuple] = None
        self._active_drawn = 0
        self.renders = 0
        self.active_rebuilds = 0

    def layer_for(self, canvas: DrawingCanvas, width: int, height: int) -> np.ndarray:
        key = (canvas.version, width, height)
        if key != self._key or self._layer is None:
            self._layer = render_strokes(canvas.strokes, width, height)
            self._key = key
            self._active_signature = None
            self.renders += 1
        active = canvas.active_stroke
        if active is None:
            self._active_layer, self._active_signature, self._active_drawn = None, None, 0
            return self._layer
        signature = (key, active.points[0], active.color, active.thickness, active.erase)
        count = len(active.points)
        if signature != self._active_signature or count < self._active_drawn or self._active_layer is None:
            self._active_layer = self._layer.copy()
            draw_stroke(self._active_layer, active)
            self.active_rebuilds += 1
        elif count > self._active_drawn:
            # Start at the last drawn point so the new segments join the old ones.
            tail = Stroke(active.points[self._active_drawn - 1:], active.color, active.thickness, active.erase)
            draw_stroke(self._active_layer, tail)
        self._active_signature, self._active_drawn = signature, count
        return self._active_layer


def export_size(aspect_w: int, aspect_h: int, long_side: int) -> tuple[int, int]:
    if aspect_w >= aspect_h:
        return long_side, max(1, round(long_side * aspect_h / aspect_w))
    return max(1, round(long_side * aspect_w / aspect_h)), long_side


def export_image(
    strokes: Iterable[Stroke], size: tuple[int, int], background: Optional[RGB] = None
) -> Image.Image:
    """Render strokes only (never camera pixels) to a PIL image."""
    width, height = size
    layer = render_strokes(strokes, width, height)
    alpha = layer[..., 3]
    if background is None:
        rgb = layer[..., :3].astype(np.uint16)
        a = alpha.astype(np.uint16)
        straight = np.where(a[..., None] > 0, np.minimum(255, rgb * 255 // np.maximum(a, 1)[..., None]), 0)
        rgba = np.dstack([straight.astype(np.uint8), alpha])
        return Image.fromarray(rgba, "RGBA")
    base = np.empty((height, width, 3), dtype=np.uint8)
    base[:] = background
    return Image.fromarray(np.ascontiguousarray(composite(base, layer)), "RGB")


def unique_export_path(directory: Path, now: Optional[_dt.datetime] = None) -> Path:
    stamp = (now or _dt.datetime.now()).strftime("%Y%m%d_%H%M%S")
    path = directory / f"handflow_{stamp}.png"
    n = 2
    while path.exists():
        path = directory / f"handflow_{stamp}_{n}.png"
        n += 1
    return path


def save_png(
    canvas: DrawingCanvas,
    directory: Path,
    aspect: tuple[int, int],
    long_side: int = 1600,
    background: Optional[RGB] = None,
) -> Path:
    """Export committed and in-progress strokes to a new PNG in ``directory``."""
    strokes = list(canvas.strokes)
    if canvas.active_stroke is not None:
        strokes.append(canvas.active_stroke)
    directory.mkdir(parents=True, exist_ok=True)
    path = unique_export_path(directory)
    export_image(strokes, export_size(*aspect, long_side), background).save(path, format="PNG")
    return path
