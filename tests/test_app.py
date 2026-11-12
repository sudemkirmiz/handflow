import time

import numpy as np
import pytest

from core.camera import CameraState
from core.finger_counter import Finger, count_fingers
from core.processing import FrameSnapshot, TrackingState
from tests.test_processing import synthetic_tracked_hand

ctk = pytest.importorskip("customtkinter")


class FakeSource:
    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.history = []
        self.draw_landmarks_calls = []

    @property
    def snapshot(self):
        return self.history[-1] if self.history else None

    @snapshot.setter
    def snapshot(self, snap):
        self.history.append(snap)

    def start(self):
        self.started += 1

    def stop(self, timeout=3.0):
        self.stopped += 1

    def latest(self):
        return self.snapshot

    def snapshots_since(self, sequence):
        return [s for s in self.history if s.sequence > sequence]

    def set_draw_landmarks(self, enabled):
        self.draw_landmarks_calls.append(enabled)


@pytest.fixture(autouse=True)
def tolerant_stale_limit(monkeypatch):
    """A busy Windows desktop can stall a freshly shown window for a moment.
    Tests that check staleness use 5 s old snapshots, so 2 s keeps them valid
    while preventing false 'stale' results in the other UI tests."""
    import config

    monkeypatch.setattr(config, "STALE_RESULT_SECONDS", 2.0)


@pytest.fixture
def app(monkeypatch):
    from ui.app import HandFlowApp

    # On Windows, CustomTkinter's constructor withdraws, update()s and re-shows
    # the window to recolor the title bar. That re-show pops the window over
    # the user's screen and steals focus, and the forced update() during
    # construction crashed Tk (access violation) after several windows had been
    # created in one process. Tests don't need a colored title bar.
    monkeypatch.setattr(HandFlowApp, "_deactivate_windows_window_header_manipulation", True)
    source = FakeSource()
    # Re-creating Tk() repeatedly in one process intermittently fails on some
    # Windows Python builds ("tcl_findLibrary"); the real app creates it once.
    error = None
    for _ in range(3):
        try:
            instance = HandFlowApp(source)
            break
        except Exception as exc:
            error = exc
    else:  # No display available (e.g. headless CI).
        pytest.skip(f"Tk not available: {error}")
    # Invisible, click-through and never activated: tests must not pop over,
    # intercept clicks or steal focus from whatever the user is doing.
    # Widgets are still laid out normally.
    instance.attributes("-alpha", 0.0)
    instance.overrideredirect(True)
    import sys

    if sys.platform.startswith("win"):
        from tools.offscreen_capture import make_ghost

        make_ghost(instance)
    yield instance, source
    instance.shutdown()


def pump(app, seconds=0.15):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.update()
        time.sleep(0.01)


def snapshot(seq, hand=None, frame=True, created_at=None):
    return FrameSnapshot(
        sequence=seq,
        created_at=created_at if created_at is not None else time.monotonic(),
        camera_state=CameraState.CONNECTED,
        camera_message="Connected · 640×480",
        tracking_state=TrackingState.TRACKING if hand else TrackingState.SEARCHING,
        frame_rgb=np.zeros((480, 640, 3), np.uint8) if frame else None,
        hand=hand,
        fingers=count_fingers([(p.x, p.y, p.z) for p in hand.world_landmarks]) if hand else None,
        fps=29.7,
    )


def test_shutdown_is_safe_and_idempotent(app):
    instance, source = app
    instance.start()
    pump(instance)
    assert instance._poll_job is not None
    instance.shutdown()
    instance.shutdown()
    assert source.stopped == 1
    assert instance._poll_job is None
    instance._poll()  # A late callback after shutdown must do nothing.
    assert instance._poll_job is None


def test_renders_hand_then_clears_on_loss(app):
    instance, source = app
    instance.start()
    source.snapshot = snapshot(1, synthetic_tracked_hand({Finger.INDEX, Finger.MIDDLE}))
    pump(instance)
    assert instance._count_label.cget("text") == "2"
    assert instance._hand_value.cget("text").startswith("Right")
    assert instance._landmarks_value.cget("text") == "21"

    source.snapshot = snapshot(2, hand=None)
    pump(instance)
    assert instance._count_label.cget("text") == "—"
    assert instance._hand_value.cget("text") == "—"


def test_stale_result_is_cleared(app):
    instance, source = app
    instance.start()
    source.snapshot = snapshot(1, synthetic_tracked_hand(set(Finger)), created_at=time.monotonic())
    pump(instance, 0.05)
    assert instance._count_label.cget("text") == "5"
    source.snapshot = snapshot(1, synthetic_tracked_hand(set(Finger)), created_at=time.monotonic() - 5)
    pump(instance, 0.05)
    assert instance._count_label.cget("text") == "—"


def test_camera_unavailable_is_shown(app):
    instance, source = app
    instance.start()
    source.snapshot = FrameSnapshot(
        sequence=1,
        created_at=time.monotonic(),
        camera_state=CameraState.UNAVAILABLE,
        camera_message="Camera unavailable — test",
        tracking_state=TrackingState.WAITING_FOR_CAMERA,
    )
    pump(instance)
    assert instance._camera_label.cget("text") == "Camera unavailable"
    placeholder = instance._preview.itemcget(instance._placeholder, "text")
    assert placeholder == "Camera unavailable — test"  # Full explanation.
    assert instance._tracking_label.cget("text") == "Waiting for camera"
