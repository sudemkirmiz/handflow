import threading
import time

import numpy as np
import pytest

from core.camera import CameraState
from core.finger_counter import Finger
from core.hand_tracker import HandTrackingResult, Landmark, TrackedHand
from core.processing import FpsCounter, FrameProcessor, TrackingState
from tests.synthetic_hand import make_hand


def synthetic_tracked_hand(extended):
    world = make_hand(extended)
    image = [Landmark(0.5 + p[0] * 2, 0.8 - p[1] * 2, p[2]) for p in world]
    return TrackedHand(
        landmarks=tuple(image),
        world_landmarks=tuple(Landmark(*p) for p in world),
        handedness="Right",
        handedness_score=0.9,
    )


class FakeCamera:
    def __init__(self, can_open=False):
        self.can_open = can_open
        self.open_calls = 0
        self.release_calls = 0
        self._open = False
        self.state = CameraState.DISCONNECTED
        self.message = "Camera not started"

    @property
    def is_open(self):
        return self._open

    def open(self):
        self.open_calls += 1
        self._open = self.can_open
        self.state = CameraState.CONNECTED if self.can_open else CameraState.UNAVAILABLE
        self.message = "Connected · 640×480" if self.can_open else "Camera unavailable"
        return self.can_open

    def read(self):
        time.sleep(0.005)
        return np.zeros((480, 640, 3), np.uint8) if self._open else None

    def release(self):
        # Like Camera.release(): idempotent, only counts real device releases.
        if self._open:
            self.release_calls += 1
        self._open = False


class FakeTracker:
    def __init__(self, hands_sequence):
        self.hands_sequence = list(hands_sequence)
        self.closed = 0
        self.ts = 0

    def detect(self, frame_rgb):
        self.ts += 1
        hand = self.hands_sequence.pop(0) if self.hands_sequence else None
        return HandTrackingResult(timestamp_ms=self.ts, hands=(hand,) if hand else ())

    def close(self):
        self.closed += 1


def frame():
    return np.zeros((480, 640, 3), np.uint8)


def test_hand_loss_clears_hand_and_finger_info_immediately():
    hand = synthetic_tracked_hand({Finger.INDEX, Finger.MIDDLE})
    tracker = FakeTracker([hand, None])
    proc = FrameProcessor(FakeCamera(can_open=True), lambda: tracker)
    proc.load_tracker()

    first = proc.process_frame(frame())
    assert first.tracking_state is TrackingState.TRACKING
    assert first.fingers.extended_count == 2
    assert first.hand.handedness == "Right"

    second = proc.process_frame(frame())
    assert second.tracking_state is TrackingState.SEARCHING
    assert second.hand is None and second.fingers is None
    assert proc.latest() is second
    assert second.sequence > first.sequence


def test_missing_model_keeps_preview_running():
    def broken_factory():
        raise FileNotFoundError("Hand Landmarker model not found")

    proc = FrameProcessor(FakeCamera(can_open=True), broken_factory)
    proc.load_tracker()
    snap = proc.process_frame(frame())
    assert snap.tracking_state is TrackingState.MODEL_UNAVAILABLE
    assert snap.frame_rgb is not None
    assert "not found" in snap.detail


def test_tracker_exception_is_reported_not_raised():
    class Exploding(FakeTracker):
        def detect(self, frame_rgb):
            raise RuntimeError("inference failed")

    proc = FrameProcessor(FakeCamera(can_open=True), lambda: Exploding([]))
    proc.load_tracker()
    snap = proc.process_frame(frame())
    assert snap.tracking_state is TrackingState.ERROR
    assert snap.hand is None


def test_frame_is_mirrored_before_detection():
    seen = {}

    class Recorder(FakeTracker):
        def detect(self, frame_rgb):
            seen["frame"] = frame_rgb.copy()
            return super().detect(frame_rgb)

    bgr = np.zeros((4, 6, 3), np.uint8)
    bgr[:, 0] = (255, 0, 0)  # Blue column on the left (BGR).
    proc = FrameProcessor(FakeCamera(can_open=True), lambda: Recorder([]), mirror=True)
    proc.load_tracker()
    proc.process_frame(bgr)
    assert tuple(seen["frame"][0, -1]) == (0, 0, 255)  # Now on the right, and RGB.


def test_no_camera_worker_retries_without_busy_loop_and_stops_cleanly():
    cam = FakeCamera(can_open=False)
    proc = FrameProcessor(cam, None, retry_seconds=0.2)
    proc.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        snap = proc.latest()
        if snap and snap.camera_state is CameraState.UNAVAILABLE:
            break
        time.sleep(0.01)
    time.sleep(0.3)
    assert proc.latest().camera_state is CameraState.UNAVAILABLE
    assert proc.latest().tracking_state is TrackingState.WAITING_FOR_CAMERA
    assert cam.open_calls <= 4  # Waits between attempts instead of spinning.

    t0 = time.monotonic()
    proc.stop()
    assert time.monotonic() - t0 < 1.0  # The retry wait is interruptible.
    assert not proc.is_running
    assert not cam.is_open


def test_stop_releases_camera_and_closes_model_once():
    cam = FakeCamera(can_open=True)
    tracker = FakeTracker([])
    proc = FrameProcessor(cam, lambda: tracker)
    proc.start()
    deadline = time.monotonic() + 2
    while (proc.latest() is None or proc.latest().frame_rgb is None) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert proc.latest().frame_rgb is not None

    proc.stop()
    proc.stop()
    assert not proc.is_running
    assert not cam.is_open
    assert cam.release_calls == 1
    assert tracker.closed == 1
    assert threading.active_count() >= 1
    proc.start()  # Restarting a stopped processor is a no-op.
    assert not proc.is_running


def test_fps_counter():
    times = iter([0.0, 0.1, 0.2, 0.3])
    fps = FpsCounter(smoothing=0.0, clock=lambda: next(times))
    assert fps.tick() is None
    assert fps.tick() == pytest.approx(10.0)
    assert fps.tick() == pytest.approx(10.0)
    fps.reset()
    assert fps.fps is None
