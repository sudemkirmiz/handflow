"""Builds synthetic 21-point hands in 3D (meters) for testing finger counting.

Coordinate frame: wrist at the origin, palm in the x-y plane, fingers point
toward +y and curl toward +z. Geometry approximates an adult right hand.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from core.finger_counter import Finger

# Long fingers: MCP position and segment lengths (proximal, middle, distal).
_LONG_FINGERS = {
    Finger.INDEX: ((0.030, 0.090, 0.0), (0.040, 0.025, 0.020), (5, 6, 7, 8)),
    Finger.MIDDLE: ((0.010, 0.095, 0.0), (0.045, 0.028, 0.022), (9, 10, 11, 12)),
    Finger.RING: ((-0.010, 0.090, 0.0), (0.042, 0.026, 0.021), (13, 14, 15, 16)),
    Finger.PINKY: ((-0.030, 0.080, 0.0), (0.030, 0.020, 0.018), (17, 18, 19, 20)),
}

EXTENDED_BENDS = (0.0, 0.0, 0.0)
FOLDED_BENDS = (70.0, 100.0, 70.0)
HALF_BENT = (10.0, 30.0, 30.0)


def _direction(theta_deg: float) -> np.ndarray:
    t = math.radians(theta_deg)
    return np.array([0.0, math.cos(t), math.sin(t)])


def make_hand(
    extended: Iterable[Finger],
    half_bent: Iterable[Finger] = (),
) -> np.ndarray:
    extended, half_bent = set(extended), set(half_bent)
    pts = np.zeros((21, 3))

    # Thumb
    cmc = np.array([0.025, 0.025, 0.0])
    pts[1] = cmc
    if Finger.THUMB in extended:
        d = np.array([0.85, 0.5, 0.0])
        d /= np.linalg.norm(d)
        pts[2] = cmc + 0.035 * d
        pts[3] = pts[2] + 0.030 * d
        pts[4] = pts[3] + 0.025 * d
    else:  # Folded across the palm.
        pts[2] = (0.045, 0.045, 0.010)
        pts[3] = (0.030, 0.060, 0.030)
        pts[4] = (0.005, 0.060, 0.030)

    for finger, (mcp, lengths, idx) in _LONG_FINGERS.items():
        if finger in extended:
            bends = EXTENDED_BENDS
        elif finger in half_bent:
            bends = HALF_BENT
        else:
            bends = FOLDED_BENDS
        pts[idx[0]] = mcp
        theta = 0.0
        for k in range(3):
            theta += bends[k]
            pts[idx[k + 1]] = pts[idx[k]] + lengths[k] * _direction(theta)
    return pts


def make_posed_hand(
    long_bends: dict[Finger, tuple[float, float, float]],
    thumb_bends: tuple[float, float] = (0.0, 0.0),
    thumb_spread: float = 30.0,
) -> np.ndarray:
    """Hand with explicit joint flexion in degrees.

    ``long_bends`` gives (MCP, PIP, DIP) flexion per long finger (missing = straight).
    ``thumb_bends`` gives (MCP, IP) flexion of the thumb, curling towards the palm.
    ``thumb_spread`` is the angle between the thumb and the index direction:
    about 30° for a relaxed open hand, 60° for a wide spread.
    """
    pts = make_hand(set(Finger))
    for finger, (mcp, lengths, idx) in _LONG_FINGERS.items():
        bends = long_bends.get(finger, EXTENDED_BENDS)
        pts[idx[0]] = mcp
        theta = 0.0
        for k in range(3):
            theta += bends[k]
            pts[idx[k + 1]] = pts[idx[k]] + lengths[k] * _direction(theta)
    # Thumb: base direction in the palm plane, flexion curls it towards +z (palm side)
    # and towards the index side.
    s = math.radians(90.0 - thumb_spread)
    base = np.array([math.cos(s), math.sin(s), 0.0])
    across = np.array([-math.sin(s), math.cos(s), 0.0])  # In-plane, towards the fingers.
    palm = np.array([0.0, 0.0, 1.0])
    cmc = np.array([0.025, 0.025, 0.0])
    pts[1] = cmc
    pts[2] = cmc + 0.035 * base
    angle = 0.0
    for k, length in enumerate((0.030, 0.025)):
        angle += math.radians(thumb_bends[k])
        d = math.cos(angle) * base + math.sin(angle) * (0.6 * palm + 0.8 * across)
        pts[3 + k] = pts[2 + k] + length * d / np.linalg.norm(d)
    return pts


def add_noise(points: np.ndarray, rng: np.random.Generator, sigma_xy: float = 0.002,
              sigma_z: float = 0.004) -> np.ndarray:
    """MediaPipe-like world landmark noise (meters); depth is noisier than x/y."""
    noise = rng.normal(0.0, 1.0, points.shape) * np.array([sigma_xy, sigma_xy, sigma_z])
    return points + noise


def rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    ax, ay, az = (math.radians(a) for a in (rx, ry, rz))
    mx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    my = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    mz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    return mz @ my @ mx


def rotate(points: np.ndarray, rx: float, ry: float, rz: float) -> np.ndarray:
    return points @ rotation(rx, ry, rz).T
