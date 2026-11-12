"""Background frame pipeline: camera → mirror → hand tracking → finger counting.

Runs in a worker thread so blocking camera reads and model inference never
freeze the UI. The UI only reads the latest immutable ``FrameSnapshot``.
"""

from __future__ import annotations

import threading
from collections import deque
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import cv2
import numpy as np

import config

from core.camera import Camera, CameraState
from core.finger_counter import FingerCountResult, FingerTracker
from core.hand_tracker import HandTracker, TrackedHand
from core.overlay import draw_hand


class TrackingState(Enum):
    WAITING_FOR_CAMERA = "Waiting for camera"
    MODEL_UNAVAILABLE = "Model unavailable"
    SEARCHING = "No hand detected"
    TRACKING = "Tracking"
    ERROR = "Tracking error"


@dataclass(frozen=True)
class FrameSnapshot:
    sequence: int
    created_at: float
    camera_state: CameraState
    camera_message: str
    tracking_state: TrackingState
    frame_rgb: Optional[np.ndarray] = None  # Mirrored frame with overlay drawn.
    hand: Optional[TrackedHand] = None  # Model output.
    fingers: Optional[FingerCountResult] = None  # Derived from the landmarks.
    fps: Optional[float] = None
    detail: str = ""


class FpsCounter:
    """Exponential moving average of processed frames per second."""

    def __init__(self, smoothing: float = 0.9, clock: Callable[[], float] = time.perf_counter) -> None:
        self._smoothing = smoothing
        self._clock = clock
        self._last: Optional[float] = None
        self.fps: Optional[float] = None

    def tick(self) -> Optional[float]:
        now = self._clock()
        if self._last is not None and now > self._last:
            instant = 1.0 / (now - self._last)
            self.fps = instant if self.fps is None else (
                self._smoothing * self.fps + (1 - self._smoothing) * instant
            )
        self._last = now
        return self.fps

    def reset(self) -> None:
        self._last = None
        self.fps = None


def finger_input_points(hand: TrackedHand, frame_shape: tuple[int, ...]) -> list[tuple[float, float, float]]:
    """Prefer metric world landmarks; otherwise undo the image aspect-ratio distortion."""
    if len(hand.world_landmarks) == 21:
        return [(p.x, p.y, p.z) for p in hand.world_landmarks]
    h, w = frame_shape[:2]
    return [(p.x * w, p.y * h, p.z * w) for p in hand.landmarks]


TrackerFactory = Callable[[], HandTracker]


class FrameProcessor:
    def __init__(
        self,
        camera: Camera,
        tracker_factory: Optional[TrackerFactory],
        mirror: bool = True,
        retry_seconds: float = 2.0,
        history_size: int = 8,
    ) -> None:
        self._camera = camera
        self._tracker_factory = tracker_factory
        self._tracker: Optional[HandTracker] = None
        self._tracker_error = ""
        self._mirror = mirror
        self._retry_seconds = retry_seconds
        self._fps = FpsCounter()
        # Stateful finger analysis (smoothing + hysteresis) for the tracked hand.
        self._fingers = FingerTracker(
            smoothing=config.FINGER_SMOOTHING,
            hysteresis_bend=config.FINGER_HYSTERESIS_BEND,
            hysteresis_reach=config.FINGER_HYSTERESIS_REACH,
        )
        self._finger_source: Optional[str] = None
        self._lock = threading.Lock()
        self._latest: Optional[FrameSnapshot] = None
        # Short ordered history so a consumer can process every tracking result
        # exactly once, even if the UI polls slower than the camera for a moment.
        self._history: deque[FrameSnapshot] = deque(maxlen=history_size)
        self.draw_landmarks = True  # Read by the worker; a bool assignment is atomic.
        self._sequence = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stopped = False

    # --- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None or self._stopped:
            return
        self._thread = threading.Thread(target=self._run, name="handflow-frames", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the worker, release the camera and close the model. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        worker_finished = self._thread is None or not self._thread.is_alive()
        self._camera.release()
        # Never close the model underneath a worker still stuck inside detect();
        # the daemon thread and its native resources end with the process.
        if self._tracker is not None and worker_finished:
            self._tracker.close()
            self._tracker = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest(self) -> Optional[FrameSnapshot]:
        with self._lock:
            return self._latest

    def snapshots_since(self, sequence: int) -> list[FrameSnapshot]:
        """Snapshots newer than ``sequence``, oldest first (bounded history)."""
        with self._lock:
            return [s for s in self._history if s.sequence > sequence]

    def set_draw_landmarks(self, enabled: bool) -> None:
        self.draw_landmarks = enabled

    # --- worker ----------------------------------------------------------------
    def load_tracker(self) -> None:
        if self._tracker is not None:
            return
        if self._tracker_factory is None:
            self._tracker_error = "No hand tracking model configured"
            return
        try:
            self._tracker = self._tracker_factory()
            self._tracker_error = ""
        except Exception as exc:  # Missing/corrupt model must not kill the preview.
            self._tracker = None
            self._tracker_error = str(exc)

    def _run(self) -> None:
        self.load_tracker()
        while not self._stop_event.is_set():
            if not self._camera.is_open:
                self._publish_status(CameraState.DISCONNECTED, "Connecting to camera…")
                opened = self._camera.open()  # Native call; may block for a while.
                if self._stop_event.is_set():
                    break  # Shutdown happened meanwhile; release below.
                if not opened:
                    self._fps.reset()
                    self._publish_status(self._camera.state, self._camera.message)
                    self._stop_event.wait(self._retry_seconds)
                    continue
            frame = self._camera.read()
            if frame is None:
                if not self._camera.is_open:  # Too many failures: camera lost.
                    self._fps.reset()
                    self._publish_status(self._camera.state, self._camera.message)
                    self._stop_event.wait(self._retry_seconds)
                else:
                    self._stop_event.wait(0.01)  # Brief back-off, no busy loop.
                continue
            self.process_frame(frame)
        # The worker owns the device while running; release it on the way out
        # (covers an open() that completed after stop() already released).
        self._camera.release()

    def process_frame(self, frame_bgr: np.ndarray) -> FrameSnapshot:
        """Process one BGR frame and publish the resulting snapshot."""
        if self._mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        hand: Optional[TrackedHand] = None
        fingers: Optional[FingerCountResult] = None
        detail = ""
        if self._tracker is None:
            state = TrackingState.MODEL_UNAVAILABLE
            detail = self._tracker_error
        else:
            try:
                result = self._tracker.detect(frame_rgb)
                hand = result.primary_hand
                state = TrackingState.TRACKING if hand else TrackingState.SEARCHING
            except Exception as exc:
                state, detail = TrackingState.ERROR, str(exc)

        if hand is None:
            self._fingers.reset()  # Hand lost: never carry state over to the next hand.
        else:
            source = "world" if len(hand.world_landmarks) == 21 else "image"
            if source != self._finger_source:
                self._fingers.reset()  # Different coordinate units: don't mix them.
                self._finger_source = source
            fingers = self._fingers.update(finger_input_points(hand, frame_rgb.shape))
            if self.draw_landmarks:
                draw_hand(frame_rgb, hand)

        snapshot = self._make_snapshot(
            camera_state=self._camera.state,
            camera_message=self._camera.message,
            tracking_state=state,
            frame_rgb=frame_rgb,
            hand=hand,
            fingers=fingers,
            fps=self._fps.tick(),
            detail=detail,
        )
        self._publish(snapshot)
        return snapshot

    def _publish_status(self, camera_state: CameraState, message: str) -> None:
        self._publish(
            self._make_snapshot(
                camera_state=camera_state,
                camera_message=message,
                tracking_state=TrackingState.WAITING_FOR_CAMERA,
            )
        )

    def _make_snapshot(self, **kwargs: object) -> FrameSnapshot:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        return FrameSnapshot(sequence=sequence, created_at=time.monotonic(), **kwargs)  # type: ignore[arg-type]

    def _publish(self, snapshot: FrameSnapshot) -> None:
        with self._lock:
            self._latest = snapshot
            self._history.append(snapshot)
