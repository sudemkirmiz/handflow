"""MediaPipe Hand Landmarker (Tasks API, VIDEO mode) integration.

Everything in ``TrackedHand`` is *model output*: landmark coordinates and the
handedness label/score come straight from MediaPipe. Derived measurements
(finger states, counts) live in ``core.finger_counter``.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

NUM_LANDMARKS = 21

# Official 21-point hand topology (matches
# mediapipe.tasks.python.vision.HandLandmarksConnections.HAND_CONNECTIONS).
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # Palm
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index
    (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
)


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class TrackedHand:
    """One detected hand, exactly as reported by the model."""

    # Normalized image coordinates (x, y in [0, 1] of the processed frame).
    landmarks: tuple[Landmark, ...]
    # Metric 3D coordinates (meters, origin near the hand's center).
    world_landmarks: tuple[Landmark, ...]
    # "Left" / "Right" from the user's point of view, or None if not reported.
    handedness: Optional[str]
    handedness_score: Optional[float]


@dataclass(frozen=True)
class HandTrackingResult:
    timestamp_ms: int
    hands: tuple[TrackedHand, ...] = field(default_factory=tuple)

    @property
    def primary_hand(self) -> Optional[TrackedHand]:
        return self.hands[0] if self.hands else None


class MonotonicTimestamper:
    """Produces strictly increasing millisecond timestamps for VIDEO mode.

    MediaPipe raises if a timestamp repeats or goes backwards, which can happen
    when two frames are processed within the same millisecond.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._start = clock()
        self._last: Optional[int] = None

    def next(self) -> int:
        now_ms = int((self._clock() - self._start) * 1000)
        if self._last is not None and now_ms <= self._last:
            now_ms = self._last + 1
        self._last = now_ms
        return now_ms


def resolve_handedness(label: Optional[str], input_is_mirrored: bool) -> Optional[str]:
    """Return handedness from the user's point of view.

    MediaPipe assumes the input image is mirrored (selfie view). HandFlow
    mirrors frames *before* detection, so the label is used as-is. If an
    un-mirrored frame is fed instead, the label must be swapped.
    """
    if label not in ("Left", "Right"):
        return None
    if input_is_mirrored:
        return label
    return "Right" if label == "Left" else "Left"


def verify_model_file(model_path: Path, expected_sha256: Optional[str]) -> None:
    """Raise if the model is missing or its SHA-256 differs from the documented one."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Hand Landmarker model not found: {path}")
    if expected_sha256:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest.lower() != expected_sha256.lower():
            raise ValueError(f"Model checksum mismatch for {path.name} — re-download it (see models/README.md)")


def _to_landmarks(points: Sequence[Any]) -> tuple[Landmark, ...]:
    return tuple(Landmark(float(p.x), float(p.y), float(p.z)) for p in points)


class HandTracker:
    """Runs the Hand Landmarker on RGB frames with monotonic timestamps."""

    def __init__(
        self,
        landmarker: Any,
        input_is_mirrored: bool = True,
        timestamper: Optional[MonotonicTimestamper] = None,
    ) -> None:
        self._landmarker = landmarker
        self.input_is_mirrored = input_is_mirrored
        self._timestamper = timestamper or MonotonicTimestamper()
        self._closed = False

    @classmethod
    def from_model(
        cls,
        model_path: Path,
        num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        input_is_mirrored: bool = True,
        expected_sha256: Optional[str] = None,
    ) -> "HandTracker":
        verify_model_file(model_path, expected_sha256)
        # Imported lazily so finger counting and tests never need MediaPipe loaded.
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)
        return cls(landmarker, input_is_mirrored=input_is_mirrored)

    def detect(self, frame_rgb: np.ndarray) -> HandTrackingResult:
        if self._closed:
            raise RuntimeError("HandTracker is closed")
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        timestamp_ms = self._timestamper.next()
        raw = self._landmarker.detect_for_video(image, timestamp_ms)
        return self.convert_result(raw, timestamp_ms)

    def convert_result(self, raw: Any, timestamp_ms: int) -> HandTrackingResult:
        """Convert a MediaPipe result into HandFlow's plain dataclasses."""
        hands: list[TrackedHand] = []
        hand_landmarks = list(getattr(raw, "hand_landmarks", None) or [])
        world = list(getattr(raw, "hand_world_landmarks", None) or [])
        handedness = list(getattr(raw, "handedness", None) or [])

        for i, points in enumerate(hand_landmarks):
            if len(points) != NUM_LANDMARKS:
                continue  # Never pass partial skeletons downstream.
            label: Optional[str] = None
            score: Optional[float] = None
            if i < len(handedness) and handedness[i]:
                top = handedness[i][0]
                label = resolve_handedness(getattr(top, "category_name", None), self.input_is_mirrored)
                score = float(top.score) if getattr(top, "score", None) is not None else None
            world_points = world[i] if i < len(world) and len(world[i]) == NUM_LANDMARKS else []
            hands.append(
                TrackedHand(
                    landmarks=_to_landmarks(points),
                    world_landmarks=_to_landmarks(world_points),
                    handedness=label,
                    handedness_score=score,
                )
            )
        return HandTrackingResult(timestamp_ms=timestamp_ms, hands=tuple(hands))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._landmarker, "close", None)
        if callable(close):
            close()
