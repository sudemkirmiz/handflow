"""OpenCV drawing of tracking results onto RGB frames (no UI toolkit involved)."""

from __future__ import annotations

import cv2
import numpy as np

from config import COLORS, hex_to_rgb
from core.hand_tracker import HAND_CONNECTIONS, TrackedHand

# Restrained styling: accent bones, light joints with a dark rim so they stay
# readable on any background; fingertips slightly larger and accent-filled.
_BONE = hex_to_rgb(COLORS["accent"])
_JOINT = hex_to_rgb(COLORS["text"])
_RIM = hex_to_rgb(COLORS["background"])
_TIP = hex_to_rgb(COLORS["accent"])
_FINGERTIPS = {4, 8, 12, 16, 20}


def draw_hand(frame_rgb: np.ndarray, hand: TrackedHand) -> None:
    """Draw the 21 landmarks and skeleton in place."""
    h, w = frame_rgb.shape[:2]
    scale = max(1, round(min(w, h) / 240))
    pts = [(int(round(p.x * w)), int(round(p.y * h))) for p in hand.landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame_rgb, pts[start], pts[end], _BONE, 2 * scale, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        tip = i in _FINGERTIPS
        radius = (3 if tip else 2) * scale + 1
        cv2.circle(frame_rgb, pt, radius + scale, _RIM, -1, cv2.LINE_AA)
        cv2.circle(frame_rgb, pt, radius, _TIP if tip else _JOINT, -1, cv2.LINE_AA)
