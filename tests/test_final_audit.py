"""Final-audit regression tests (no GUI)."""

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import config
from core.air_canvas import EXECUTED_LOG_LIMIT, Action, AirCanvasController, Command
from core.camera import CameraState
from core.canvas import DrawingCanvas
from core.canvas_render import save_png
from core.processing import FrameProcessor
from tests.gesture_helpers import DRAW_FINGERS, FIST_FINGERS, UNCERTAIN_INDEX, Feeder

# --- configuration integrity -----------------------------------------------------------------------
# Pinned values. A UI-redesign script once deleted the Air Canvas block of
# config.py by accident; it was restored and verified byte-for-byte against the
# pre-incident checkpoint. This test makes any future silent change visible.
# Change a value here only together with a deliberate, documented config change.
PINNED = {
    # Camera / MediaPipe
    "CAMERA_INDEX": 0, "CAMERA_WIDTH": 640, "CAMERA_HEIGHT": 480, "MIRROR_PREVIEW": True,
    "MAX_CONSECUTIVE_READ_FAILURES": 15, "CAMERA_RETRY_SECONDS": 2.0, "MAX_HANDS": 1,
    "MIN_HAND_DETECTION_CONFIDENCE": 0.6, "MIN_HAND_PRESENCE_CONFIDENCE": 0.5, "MIN_TRACKING_CONFIDENCE": 0.5,
    "MODEL_SHA256": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    # Finger analysis over time
    "FINGER_SMOOTHING": 0.35, "FINGER_HYSTERESIS_BEND": 15.0, "FINGER_HYSTERESIS_REACH": 0.07,
    # UI
    "UI_POLL_MS": 15, "STALE_RESULT_SECONDS": 0.5, "WINDOW_SIZE": "1180x720", "WINDOW_MIN_SIZE": (960, 600),
    # Brushes, canvas, export
    "BRUSH_COLORS": {"Pink": "#FF8FCF", "Purple": "#B58CFF", "Blue": "#7FB8FF", "Mint": "#8EF0C4",
                     "Yellow": "#FFE38F", "White": "#FFFFFF"},
    "BRUSH_SIZES": {"Thin": 0.008, "Medium": 0.016, "Thick": 0.030},
    "ERASER_SIZE_FACTOR": 2.5, "EXPORT_LONG_SIDE": 1600,
    "EXPORT_BACKGROUNDS": {"Transparent": None, "White": "#FFFFFF", "Dark": "#111318"},
    "MAX_STROKES": 2000, "MAX_POINTS_PER_STROKE": 4000, "UNDO_LIMIT": 50, "MIN_POINT_DISTANCE": 0.002,
    # Gestures, dwell, cooldowns
    "GESTURE_HOLD_SECONDS": 0.12, "SELECT_HOLD_SECONDS": 0.18, "DWELL_SECONDS": 0.5,
    "ACTION_DWELL_SECONDS": 1.0, "HOVER_MAX_SPEED": 0.6, "COMMAND_COOLDOWN_SECONDS": 0.6,
    "SAVE_CLEAR_COOLDOWN_SECONDS": 2.0, "CLEAR_CONFIRM_SECONDS": 6.0, "STROKE_MAX_GAP_SECONDS": 0.5,
    "STROKE_MAX_JUMP": 0.15, "UNCERTAIN_GRACE_SECONDS": 0.35, "DROPOUT_RESUME_SECONDS": 0.15,
    "RESUME_MAX_DISTANCE": 0.05, "DWELL_GRACE_SECONDS": 0.3, "HOVER_MARGIN": 0.02,
    # Pointer smoothing
    "POINTER_MIN_CUTOFF_HZ": 1.0, "POINTER_BETA": 10.0,
    "TOOLBAR_POINTER_MIN_CUTOFF_HZ": 0.5, "TOOLBAR_POINTER_BETA": 3.0,
    "DIAGNOSTICS_DEFAULT": False,
}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_config_value_is_pinned(name):
    assert getattr(config, name) == PINNED[name]


def test_default_brush_is_pink_medium():
    ctl = AirCanvasController(DrawingCanvas())
    assert (ctl.color_name, ctl.size_name) == ("Pink", "Medium")


# --- camera recovery ----------------------------------------------------------------------------------

class FlakyCamera:
    """Unavailable for the first opens, then delivers frames (e.g. replugged USB)."""

    def __init__(self, failures):
        self.failures = failures
        self.open_calls = 0
        self._open = False
        self.state = CameraState.DISCONNECTED
        self.message = ""

    @property
    def is_open(self):
        return self._open

    def open(self):
        self.open_calls += 1
        self._open = self.open_calls > self.failures
        self.state = CameraState.CONNECTED if self._open else CameraState.UNAVAILABLE
        self.message = "Connected · 640×480" if self._open else "Camera unavailable"
        return self._open

    def read(self):
        time.sleep(0.005)
        return np.zeros((48, 64, 3), np.uint8) if self._open else None

    def release(self):
        self._open = False


def test_camera_recovers_after_being_unavailable():
    camera = FlakyCamera(failures=2)
    proc = FrameProcessor(camera, None, retry_seconds=0.05)
    proc.start()
    seen = set()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        snap = proc.latest()
        if snap is not None:
            seen.add(snap.camera_state)
            if snap.frame_rgb is not None:
                break
        time.sleep(0.01)
    proc.stop()
    assert CameraState.UNAVAILABLE in seen
    assert proc.latest().frame_rgb is not None  # Frames flow again after reconnecting.
    assert camera.open_calls == 3
    assert not proc.is_running and not camera.is_open


# --- long sessions stay bounded -------------------------------------------------------------------------

def test_long_session_keeps_internal_logs_bounded():
    ctl = AirCanvasController(DrawingCanvas(), save_handler=lambda: Path("x.png"))
    for i in range(EXECUTED_LOG_LIMIT * 3):
        ctl.execute(Command(Action.COLOR, list(config.BRUSH_COLORS)[i % 6]), float(i))
    assert len(ctl.executed) == EXECUTED_LOG_LIMIT
    f = Feeder(ctl)
    for k in range(3000):  # ~100 s of mixed input.
        fingers = (DRAW_FINGERS, UNCERTAIN_INDEX, FIST_FINGERS)[(k // 40) % 3]
        f.frame(fingers, (0.3 + 0.4 * ((k % 80) / 80), 0.6))
    assert len(ctl.recent) <= 90
    assert len(ctl._pending_draw) < 10
    assert len(ctl.canvas.strokes) <= config.MAX_STROKES


# --- export ------------------------------------------------------------------------------------------------

def test_dark_export_is_a_valid_png_with_theme_background(tmp_path):
    canvas = DrawingCanvas()
    canvas.set_color((255, 143, 207))
    canvas.begin_stroke((0.2, 0.5))
    canvas.add_point((0.8, 0.5))
    canvas.end_stroke()
    dark = config.hex_to_rgb(config.EXPORT_BACKGROUNDS["Dark"])
    path = save_png(canvas, tmp_path, aspect=(640, 480), background=dark)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        assert img.mode == "RGB" and img.size == (1600, 1200)
        assert img.getpixel((5, 5)) == dark == config.hex_to_rgb(config.COLORS["background"])
        assert img.getpixel((800, 600)) == (255, 143, 207)
