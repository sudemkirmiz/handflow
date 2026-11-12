"""Finger extension analysis from 21 hand landmarks.

This module is pure geometry (NumPy only) and knows nothing about the camera,
MediaPipe or the UI, so it can be tested with synthetic landmarks.

Method
------
Each finger is judged with two rotation-invariant cues computed in 3D:

1. **Bend** — how far the joints deviate from a straight line. The angle at a
   joint is 180° when the finger is straight; ``bend = 180° - angle``.
2. **Reach** — whether the fingertip is farther from a reference point than
   the middle joint is. A curled finger brings its tip back toward the palm.

For the four long fingers the reference is the wrist. The thumb moves
sideways across the palm instead of curling toward the wrist, so its
reference is the pinky knuckle (landmark 17): a thumb folded across the palm
brings its tip closer to the pinky side.

When the two cues disagree or sit between thresholds the finger is reported
as ``UNCERTAIN`` rather than guessed. Use world landmarks when available —
normalized image coordinates are distorted by the frame's aspect ratio and
have weaker depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np

NUM_LANDMARKS = 21
WRIST = 0
PINKY_MCP = 17


class Finger(Enum):
    THUMB = "Thumb"
    INDEX = "Index"
    MIDDLE = "Middle"
    RING = "Ring"
    PINKY = "Pinky"


class FingerState(Enum):
    EXTENDED = "extended"
    FOLDED = "folded"
    UNCERTAIN = "uncertain"


# (base, joint1, joint2, tip) landmark indices per finger.
FINGER_JOINTS: dict[Finger, tuple[int, int, int, int]] = {
    Finger.THUMB: (1, 2, 3, 4),  # CMC, MCP, IP, TIP
    Finger.INDEX: (5, 6, 7, 8),  # MCP, PIP, DIP, TIP
    Finger.MIDDLE: (9, 10, 11, 12),
    Finger.RING: (13, 14, 15, 16),
    Finger.PINKY: (17, 18, 19, 20),
}


@dataclass(frozen=True)
class Thresholds:
    # Long fingers: summed bend at PIP + DIP, in degrees.
    finger_extended_max_bend: float = 50.0
    finger_folded_min_bend: float = 90.0
    # Long fingers: dist(wrist, tip) / dist(wrist, pip).
    finger_extended_min_reach: float = 1.15
    finger_folded_max_reach: float = 1.0
    # Thumb: summed bend at MCP + IP, in degrees.
    thumb_extended_max_bend: float = 45.0
    thumb_folded_min_bend: float = 75.0
    # Thumb: dist(pinky_mcp, tip) / dist(pinky_mcp, ip).
    # 1.05 (was 1.10): a relaxed open thumb measures ~1.09 even without noise,
    # while folded thumbs (fists) measure 0.72-0.82, far below either value.
    thumb_extended_min_reach: float = 1.05
    thumb_folded_max_reach: float = 1.0


@dataclass(frozen=True)
class FingerMeasurement:
    state: FingerState
    bend_degrees: Optional[float]
    reach_ratio: Optional[float]


@dataclass(frozen=True)
class FingerCountResult:
    valid: bool
    fingers: dict[Finger, FingerMeasurement]
    reason: str = ""

    @property
    def extended_count(self) -> Optional[int]:
        """Number of fingers confidently extended (0–5), or None if invalid."""
        if not self.valid:
            return None
        return sum(m.state is FingerState.EXTENDED for m in self.fingers.values())

    @property
    def uncertain_count(self) -> int:
        return sum(m.state is FingerState.UNCERTAIN for m in self.fingers.values())

    def state_of(self, finger: Finger) -> FingerState:
        return self.fingers[finger].state


def _uncertain_all(reason: str) -> FingerCountResult:
    unknown = FingerMeasurement(FingerState.UNCERTAIN, None, None)
    return FingerCountResult(valid=False, fingers={f: unknown for f in Finger}, reason=reason)


def to_array(landmarks: Optional[Sequence[Any]]) -> Optional[np.ndarray]:
    """Convert landmarks (objects with x/y/z, or (x, y, z) sequences) to a (21, 3) array."""
    if landmarks is None:
        return None
    try:
        if len(landmarks) != NUM_LANDMARKS:
            return None
        rows = [
            (p.x, p.y, p.z) if hasattr(p, "x") else tuple(p) for p in landmarks
        ]
        array = np.asarray(rows, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.shape != (NUM_LANDMARKS, 3) or not np.all(np.isfinite(array)):
        return None
    return array


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    """Angle ABC in degrees (180 = straight). None if a segment has zero length."""
    ba, bc = a - b, c - b
    na, nc = float(np.linalg.norm(ba)), float(np.linalg.norm(bc))
    if na < 1e-9 or nc < 1e-9:
        return None
    cos = float(np.dot(ba, bc) / (na * nc))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _classify(
    bend: Optional[float],
    reach: Optional[float],
    extended_max_bend: float,
    folded_min_bend: float,
    extended_min_reach: float,
    folded_max_reach: float,
) -> FingerState:
    if bend is None or reach is None:
        return FingerState.UNCERTAIN
    if bend <= extended_max_bend and reach >= extended_min_reach:
        return FingerState.EXTENDED
    if bend >= folded_min_bend or reach <= folded_max_reach:
        return FingerState.FOLDED
    return FingerState.UNCERTAIN


def _measure_finger(
    points: np.ndarray, finger: Finger, t: Thresholds
) -> FingerMeasurement:
    base, j1, j2, tip = (points[i] for i in FINGER_JOINTS[finger])
    a1 = joint_angle(base, j1, j2)
    a2 = joint_angle(j1, j2, tip)
    bend = None if a1 is None or a2 is None else (180.0 - a1) + (180.0 - a2)

    if finger is Finger.THUMB:
        ref, mid = points[PINKY_MCP], j2  # Thumb IP joint.
        limits = (
            t.thumb_extended_max_bend,
            t.thumb_folded_min_bend,
            t.thumb_extended_min_reach,
            t.thumb_folded_max_reach,
        )
    else:
        ref, mid = points[WRIST], j1  # PIP joint.
        limits = (
            t.finger_extended_max_bend,
            t.finger_folded_min_bend,
            t.finger_extended_min_reach,
            t.finger_folded_max_reach,
        )

    mid_dist = float(np.linalg.norm(mid - ref))
    reach = None if mid_dist < 1e-9 else float(np.linalg.norm(tip - ref)) / mid_dist
    return FingerMeasurement(_classify(bend, reach, *limits), bend, reach)


def count_fingers(
    landmarks: Optional[Sequence[Any]], thresholds: Thresholds = Thresholds()
) -> FingerCountResult:
    """Classify each finger as extended / folded / uncertain."""
    points = to_array(landmarks)
    if points is None:
        return _uncertain_all("Missing or invalid landmarks")
    fingers = {f: _measure_finger(points, f, thresholds) for f in Finger}
    return FingerCountResult(valid=True, fingers=fingers)


class FingerTracker:
    """Stateful finger analysis for a live video stream.

    ``count_fingers`` judges every frame on its own. On real landmarks that is
    fragile: measurement noise always *adds* apparent bend (a straight finger
    can't look straighter than straight), and five independent fingers
    compound the error, so an open hand flickers between 3, 4 and 5.

    This tracker adds two small things on top of the same geometry:

    * light exponential smoothing of the 21 landmarks (reset on hand loss or a
      sudden jump, e.g. a different hand), which removes most of the noise;
    * per-finger hysteresis: a finger that is Up stays Up until it is clearly
      bent (and vice versa), so values hovering at a threshold don't flicker.

    Clear transitions (opening a fist, closing a hand) still show up within a
    few frames.
    """

    def __init__(
        self,
        smoothing: float = 0.35,
        hysteresis_bend: float = 15.0,
        hysteresis_reach: float = 0.07,
        jump_ratio: float = 0.6,
        thresholds: Thresholds = Thresholds(),
    ) -> None:
        self.smoothing = smoothing
        self.hysteresis_bend = hysteresis_bend
        self.hysteresis_reach = hysteresis_reach
        self.jump_ratio = jump_ratio
        self.thresholds = thresholds
        self._smoothed: Optional[np.ndarray] = None
        self._previous: dict[Finger, FingerState] = {}

    def reset(self) -> None:
        self._smoothed = None
        self._previous = {}

    def update(self, landmarks: Optional[Sequence[Any]]) -> FingerCountResult:
        points = to_array(landmarks)
        if points is None:
            self.reset()
            return _uncertain_all("Missing or invalid landmarks")
        if self._smoothed is None or self._jumped(points):
            self.reset()
            self._smoothed = points.copy()
        else:
            self._smoothed += self.smoothing * (points - self._smoothed)
        fingers = {}
        for finger in Finger:
            measured = _measure_finger(self._smoothed, finger, self.thresholds)
            state = self._with_hysteresis(finger, measured)
            self._previous[finger] = state
            fingers[finger] = FingerMeasurement(state, measured.bend_degrees, measured.reach_ratio)
        return FingerCountResult(valid=True, fingers=fingers)

    def _jumped(self, points: np.ndarray) -> bool:
        palm = float(np.linalg.norm(self._smoothed[9] - self._smoothed[WRIST]))
        if palm < 1e-9:
            return True
        shift = float(np.max(np.linalg.norm(points - self._smoothed, axis=1)))
        return shift > self.jump_ratio * palm

    def _with_hysteresis(self, finger: Finger, m: FingerMeasurement) -> FingerState:
        previous = self._previous.get(finger)
        if m.bend_degrees is None or m.reach_ratio is None or previous is None:
            return m.state
        t = self.thresholds
        thumb = finger is Finger.THUMB
        ext_bend = t.thumb_extended_max_bend if thumb else t.finger_extended_max_bend
        ext_reach = t.thumb_extended_min_reach if thumb else t.finger_extended_min_reach
        fold_bend = t.thumb_folded_min_bend if thumb else t.finger_folded_min_bend
        fold_reach = t.thumb_folded_max_reach if thumb else t.finger_folded_max_reach
        hb, hr = self.hysteresis_bend, self.hysteresis_reach
        if previous is FingerState.EXTENDED and m.bend_degrees <= ext_bend + hb and m.reach_ratio >= ext_reach - hr:
            return FingerState.EXTENDED
        if previous is FingerState.FOLDED and (m.bend_degrees >= fold_bend - hb or m.reach_ratio <= fold_reach + hr):
            return FingerState.FOLDED
        return m.state
