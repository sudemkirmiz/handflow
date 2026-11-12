"""One-finger interaction and UI changes, in the real UI with mock snapshots.
The window is invisible and never takes focus (see the ``app`` fixture)."""

import time

import numpy as np
import pytest

import config
from core.air_canvas import HINT, Action, Command
from core.camera import CameraState
from core.finger_counter import Finger
from core.hand_tracker import Landmark, TrackedHand
from core.processing import FrameSnapshot, TrackingState
from tests.gesture_helpers import DRAW_FINGERS
from tests.synthetic_hand import make_hand
from tests.test_app import app, pump, tolerant_stale_limit  # noqa: F401  (pytest fixtures)

pytest.importorskip("customtkinter")


def snap(seq, u, v, t, fingers=DRAW_FINGERS):
    world = make_hand({Finger.INDEX})
    image = [Landmark(0.5, 0.5, 0.0)] * 21
    image[8] = Landmark(u, v, 0.0)
    hand = TrackedHand(tuple(image), tuple(Landmark(*p) for p in world), "Right", 0.9)
    return FrameSnapshot(seq, t, CameraState.CONNECTED, "Connected", TrackingState.TRACKING,
                         np.zeros((480, 640, 3), np.uint8), hand, fingers, 30.0)


class Feed:
    """Pushes 30 FPS snapshots ending at 'now' and lets the UI process them."""

    def __init__(self, instance, source):
        self.instance, self.source, self.seq = instance, source, 0

    def points(self, pts):
        t0 = time.monotonic() - len(pts) / 30
        for k, (u, v) in enumerate(pts):
            self.seq += 1
            self.source.snapshot = snap(self.seq, u, v, t0 + k / 30)
        self.instance.process_new_snapshots()

    def hold(self, point, seconds):
        self.points([point] * round(seconds * 30))


def center_of(ctl, action, value=""):
    item = next(i for i in ctl.toolbar if i.command == Command(action, value))
    return ((item.rect[0] + item.rect[2]) / 2, (item.rect[1] + item.rect[3]) / 2)


def test_panel_texts_and_no_duplicate_palette(app):  # noqa: F811
    from ui.app import Mode

    instance, _ = app
    instance.set_mode(Mode.AIR_CANVAS)
    panel = instance._canvas_panel
    def walk(widget):
        for child in widget.winfo_children():
            yield child
            yield from walk(child)

    texts = []
    for w in walk(panel):
        try:
            texts.append(w.cget("text"))
        except Exception:
            pass  # Frames and canvases have no text.
    assert "Export Background" in texts and "PNG BACKGROUND" not in texts
    # The instruction now sits in the workspace footer, next to the camera image.
    assert instance._hint_label.cget("text") == HINT == "☝ Point to draw · Hover over a tool to select"
    assert not hasattr(panel, "swatches")
    assert list(config.EXPORT_BACKGROUNDS) == ["Transparent", "White", "Dark"]
    assert config.EXPORT_BACKGROUNDS["Dark"] == config.COLORS["background"]
    assert panel.size_buttons.get() == "Medium"  # Default brush size.


def test_one_finger_selects_color_in_the_real_ui(app):  # noqa: F811
    from ui.app import Mode

    instance, source = app
    instance.set_mode(Mode.AIR_CANVAS)
    pump(instance, 0.2)
    ctl, feed = instance.controller, Feed(instance, source)
    target = center_of(ctl, Action.COLOR, "Mint")
    feed.hold(target, 0.15 + config.DWELL_SECONDS * 0.5)  # Halfway through the dwell.
    assert instance._canvas_panel.mode_value.cget("text") == "Selecting"
    arc = instance._cursor_items[2]
    assert instance._preview.itemcget(arc, "state") == "normal"  # Progress ring visible.
    assert -360 < float(instance._preview.itemcget(arc, "extent")) < 0
    feed.hold(target, config.DWELL_SECONDS)
    assert ctl.color_name == "Mint"
    assert ctl.canvas.is_empty  # Hovering never drew anything.
    assert "Mint" in instance._canvas_panel.tool_value.cget("text")


def test_one_finger_draws_then_selects_then_draws(app):  # noqa: F811
    from ui.app import Mode

    instance, source = app
    instance.set_mode(Mode.AIR_CANVAS)
    pump(instance, 0.2)
    ctl, feed = instance.controller, Feed(instance, source)
    feed.points([(0.2 + 0.01 * k, 0.6) for k in range(30)])
    assert instance._canvas_panel.mode_value.cget("text") == "Drawing"
    arc = instance._cursor_items[2]
    assert instance._preview.itemcget(arc, "state") == "hidden"  # No ring while drawing.
    blue = center_of(ctl, Action.COLOR, "Blue")
    feed.points([(0.5 + (blue[0] - 0.5) * k / 15, 0.6 + (blue[1] - 0.6) * k / 15) for k in range(1, 16)])
    feed.hold(blue, config.DWELL_SECONDS + 0.2)
    feed.points([(blue[0], blue[1] + (0.7 - blue[1]) * k / 15) for k in range(1, 16)])
    feed.points([(blue[0] + 0.01 * k, 0.7) for k in range(20)])
    first, second = ctl.canvas.strokes[0], (ctl.canvas.active_stroke or ctl.canvas.strokes[-1])
    assert first.color == config.hex_to_rgb(config.BRUSH_COLORS["Pink"])
    assert second.color == config.hex_to_rgb(config.BRUSH_COLORS["Blue"])
    assert all(p[1] > 0.15 for s in (first, second) for p in s.points)


@pytest.mark.parametrize("background,mode,pixel", [
    ("Transparent", "RGBA", (0, 0, 0, 0)),
    ("White", "RGB", (255, 255, 255)),
    ("Dark", "RGB", (17, 19, 24)),
])
def test_export_background_changes_png_only(app, tmp_path, monkeypatch, background, mode, pixel):  # noqa: F811
    from PIL import Image
    from ui.app import Mode

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    instance, _ = app
    instance.set_mode(Mode.AIR_CANVAS)
    ctl = instance.controller
    instance.run_command(Command(Action.COLOR, "Yellow"))
    ctl.mouse_press((0.3, 0.5), 0)
    ctl.mouse_drag((0.7, 0.5))
    ctl.mouse_release()
    instance._canvas_panel.background_buttons.set(background)
    instance._canvas_panel._set_background(background)
    assert ctl.color_name == "Yellow"  # Background choice never changes the brush color.
    assert instance.run_command(Command(Action.SAVE)).startswith("Saved")
    (path,) = tmp_path.glob("handflow_*.png")
    with Image.open(path) as img:
        assert img.mode == mode
        assert img.getpixel((10, 10)) == pixel
        stroke = img.getpixel((800, 600))
        assert stroke[:3] == config.hex_to_rgb(config.BRUSH_COLORS["Yellow"])
