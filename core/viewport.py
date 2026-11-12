"""Mapping between normalized frame coordinates and the on-screen preview.

The camera frame is scaled to fit the preview area while keeping its aspect
ratio, and centered. The leftover space (letterbox margins) is not part of the
image, so a point there does not map to the canvas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Viewport:
    frame_w: int
    frame_h: int
    area_w: int
    area_h: int
    display_w: int
    display_h: int
    offset_x: int
    offset_y: int

    def to_display(self, u: float, v: float) -> tuple[float, float]:
        """Normalized frame point -> preview-area pixels."""
        return (self.offset_x + u * self.display_w, self.offset_y + v * self.display_h)

    def to_image(self, u: float, v: float) -> tuple[float, float]:
        """Normalized frame point -> pixels inside the displayed (resized) image."""
        return (u * self.display_w, v * self.display_h)

    def to_normalized(self, x: float, y: float) -> Optional[tuple[float, float]]:
        """Preview-area pixels -> normalized frame point, or None in the margins."""
        u = (x - self.offset_x) / self.display_w
        v = (y - self.offset_y) / self.display_h
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
            return (u, v)
        return None


def fit_viewport(frame_w: int, frame_h: int, area_w: int, area_h: int) -> Optional[Viewport]:
    """Aspect-preserving fit of a frame into an area. None if either is empty."""
    if min(frame_w, frame_h, area_w, area_h) <= 0:
        return None
    scale = min(area_w / frame_w, area_h / frame_h)
    display_w = max(1, int(frame_w * scale))
    display_h = max(1, int(frame_h * scale))
    return Viewport(
        frame_w=frame_w,
        frame_h=frame_h,
        area_w=area_w,
        area_h=area_h,
        display_w=display_w,
        display_h=display_h,
        offset_x=(area_w - display_w) // 2,
        offset_y=(area_h - display_h) // 2,
    )


def landmark_to_canvas(
    x: float, y: float, detection_input_mirrored: bool, display_mirrored: bool
) -> tuple[float, float]:
    """Express a landmark in the coordinates of the *displayed* frame.

    HandFlow mirrors the frame before detection and displays that same frame,
    so this is the identity in the app. If detection ran on the raw frame but
    the preview is mirrored (or vice versa), x must be flipped.
    """
    if detection_input_mirrored != display_mirrored:
        return (1.0 - x, y)
    return (x, y)
