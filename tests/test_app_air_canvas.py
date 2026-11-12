"""Air Canvas inside the real CustomTkinter UI, driven by mock snapshots and tk events."""

import time

import numpy as np
import pytest

import config
from core.air_canvas import Action, Command
from core.camera import CameraState
from core.finger_counter import Finger
from core.gestures import Gesture
from core.hand_tracker import Landmark, TrackedHand
from core.processing import FrameSnapshot, TrackingState
from tests.gesture_helpers import DRAW_FINGERS
from tests.synthetic_hand import make_hand
from tests.test_app import app, pump, tolerant_stale_limit  # noqa: F401  (pytest fixtures)

pytest.importorskip("customtkinter")


def hand_pointing_at(u, v):
    """A TrackedHand whose index fingertip (landmark 8) sits at (u, v)."""
    world = make_hand({Finger.INDEX})
    image = [Landmark(0.5, 0.5, 0.0)] * 21
    image[8] = Landmark(u, v, 0.0)
    return TrackedHand(tuple(image), tuple(Landmark(*p) for p in world), "Right", 0.9)


def draw_snapshot(seq, u, v, fingers=DRAW_FINGERS, created_at=None):
    return FrameSnapshot(
        sequence=seq,
        created_at=created_at if created_at is not None else time.monotonic(),
        camera_state=CameraState.CONNECTED,
        camera_message="Connected · 640×480",
        tracking_state=TrackingState.TRACKING,
        frame_rgb=np.zeros((480, 640, 3), np.uint8),
        hand=hand_pointing_at(u, v),
        fingers=fingers,
        fps=30.0,
    )


def feed_moving_right(source, count, start_age=None):
    """30 FPS frames moving right; by default the last frame is stamped 'now'."""
    t0 = time.monotonic() - (count / 30 if start_age is None else start_age)
    for i in range(1, count + 1):
        source.snapshot = draw_snapshot(i, 0.2 + i * 0.02, 0.5, created_at=t0 + i / 30)


# --- 13. mode switching ------------------------------------------------------------------

def test_mode_switching(app):  # noqa: F811
    from ui.app import Mode

    instance, source = app
    pump(instance, 0.05)
    assert instance.mode is Mode.HAND_TRACKING
    assert instance._analysis_card.winfo_ismapped()

    instance.set_mode(Mode.AIR_CANVAS)
    pump(instance, 0.05)
    assert instance._canvas_panel.winfo_ismapped() and not instance._analysis_card.winfo_ismapped()
    assert source.draw_landmarks_calls[-1] is False  # Fingertip cursor instead of skeleton.

    instance.set_mode(Mode.HAND_TRACKING)
    pump(instance, 0.05)
    assert instance._analysis_card.winfo_ismapped() and not instance._canvas_panel.winfo_ismapped()
    assert source.draw_landmarks_calls[-1] is True
    assert instance._preview.find_withtag("overlay") == ()


def test_hand_tracking_mode_does_not_draw(app):  # noqa: F811
    instance, source = app
    feed_moving_right(source, 15)
    instance.process_new_snapshots()
    assert instance.controller.canvas.is_empty


def test_switching_mode_mid_stroke_commits_it(app):  # noqa: F811
    from ui.app import Mode

    instance, source = app
    instance.set_mode(Mode.AIR_CANVAS)
    feed_moving_right(source, 10)
    instance.process_new_snapshots()
    assert instance.controller.canvas.is_drawing
    instance.set_mode(Mode.HAND_TRACKING)
    assert not instance.controller.canvas.is_drawing
    assert len(instance.controller.canvas.strokes) == 1
    assert instance.controller.gesture is Gesture.NO_HAND


# --- ordered, de-duplicated snapshot processing ---------------------------------------------

def test_snapshots_are_processed_once_and_in_order(app):  # noqa: F811
    from ui.app import Mode

    instance, source = app
    instance.set_mode(Mode.AIR_CANVAS)
    feed_moving_right(source, 15)
    assert instance.process_new_snapshots() == 15
    assert instance.process_new_snapshots() == 0  # Nothing new: nothing re-processed.
    assert instance.controller.last_sequence == 15
    xs = [p[0] for p in instance.controller.canvas.active_stroke.points]
    assert xs == sorted(xs)


def test_stale_snapshots_lift_pen(app):  # noqa: F811
    from ui.app import Mode

    instance, source = app
    instance.set_mode(Mode.AIR_CANVAS)
    feed_moving_right(source, 10, start_age=5.0)  # Old frames, then the camera stalls.
    instance.process_new_snapshots()
    assert instance.controller.canvas.is_drawing
    instance.process_new_snapshots()  # No new frames and the latest is stale.
    assert not instance.controller.canvas.is_drawing


# --- 6/7. coordinate mapping in the real widget, resizing ------------------------------------

def test_mouse_drawing_maps_through_letterboxed_preview(app):  # noqa: F811
    from ui.app import Mode

    instance, _ = app
    instance.set_mode(Mode.AIR_CANVAS)  # No camera: a blank 640x480 canvas is shown.
    pump(instance, 0.2)
    vp = instance._viewport
    assert vp is not None and (vp.frame_w, vp.frame_h) == (640, 480)
    x0, y0 = vp.to_display(0.3, 0.5)
    x1, y1 = vp.to_display(0.7, 0.6)
    cv = instance._preview
    cv.event_generate("<ButtonPress-1>", x=round(x0), y=round(y0))
    for k in range(1, 11):
        cv.event_generate("<B1-Motion>", x=round(x0 + (x1 - x0) * k / 10), y=round(y0 + (y1 - y0) * k / 10))
    cv.event_generate("<ButtonRelease-1>", x=round(x1), y=round(y1))
    pump(instance, 0.05)
    (stroke,) = instance.controller.canvas.strokes
    assert stroke.points[0] == pytest.approx((0.3, 0.5), abs=0.01)
    assert stroke.points[-1] == pytest.approx((0.7, 0.6), abs=0.01)
    assert vp.offset_x > 2 or vp.offset_y > 2  # The preview really is letterboxed.
    # A press inside the letterbox margin is outside the image and is ignored.
    mx, my = (1, round(y0)) if vp.offset_x > 2 else (round(x0), 1)
    cv.event_generate("<ButtonPress-1>", x=mx, y=my)
    cv.event_generate("<ButtonRelease-1>", x=mx, y=my)
    assert len(instance.controller.canvas.strokes) == 1


def test_strokes_survive_window_resize(app):  # noqa: F811
    from ui.app import Mode

    instance, _ = app
    instance.set_mode(Mode.AIR_CANVAS)
    pump(instance, 0.2)
    instance.controller.mouse_press((0.3, 0.5), 0)
    instance.controller.mouse_drag((0.6, 0.5))
    instance.controller.mouse_release()
    before = instance.controller.canvas.strokes
    vp_before = instance._viewport
    instance.geometry("1000x620")
    pump(instance, 0.3)
    instance.geometry("1150x700")
    pump(instance, 0.3)
    assert instance.controller.canvas.strokes == before
    assert instance._viewport != vp_before  # The display mapping followed the new size.


# --- 8/11. commands and export through the UI --------------------------------------------------

def test_ui_commands_and_png_export(app, tmp_path, monkeypatch):  # noqa: F811
    from PIL import Image
    from ui.app import Mode

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    instance, _ = app
    instance.set_mode(Mode.AIR_CANVAS)
    instance.run_command(Command(Action.COLOR, "Mint"))
    instance.run_command(Command(Action.SIZE, "Thick"))
    instance.controller.mouse_press((0.3, 0.5), 0)
    instance.controller.mouse_drag((0.6, 0.5))
    instance.controller.mouse_release()
    instance._canvas_panel._set_background("White")
    msg = instance.run_command(Command(Action.SAVE))
    files = list(tmp_path.glob("handflow_*.png"))
    assert msg.startswith("Saved") and len(files) == 1
    with Image.open(files[0]) as img:
        assert img.mode == "RGB" and img.size == (1600, 1200)
        assert img.getpixel((int(0.45 * 1600), int(0.5 * 1200))) == config.hex_to_rgb(config.BRUSH_COLORS["Mint"])


# --- 14. responsiveness and shutdown ----------------------------------------------------------------

def test_render_stays_fast_with_many_strokes(app):  # noqa: F811
    from ui.app import Mode

    instance, source = app
    instance.set_mode(Mode.AIR_CANVAS)
    canvas = instance.controller.canvas
    for i in range(300):
        canvas.begin_stroke((0.1 + (i % 30) * 0.03, 0.2 + (i // 30) * 0.07))
        canvas.add_point((0.12 + (i % 30) * 0.03, 0.25 + (i // 30) * 0.07))
        canvas.end_stroke()
    pump(instance, 0.2)
    renders_before = instance._layer_cache.renders
    times = []
    t0 = time.monotonic()
    for i in range(1, 31):
        source.snapshot = draw_snapshot(i, 0.5, 0.4 + i * 0.005, created_at=t0 + i / 30)
        start = time.perf_counter()
        instance.process_new_snapshots()
        instance.update_idletasks()
        times.append(time.perf_counter() - start)
    median = sorted(times)[len(times) // 2]
    assert median < 0.030, f"median frame {median * 1000:.1f} ms"
    # Committed strokes are rasterized once; only the live stroke is redrawn per frame.
    assert instance._layer_cache.renders - renders_before <= 2


def test_shutdown_in_air_canvas_mid_stroke(app):  # noqa: F811
    from ui.app import Mode

    instance, source = app
    instance.set_mode(Mode.AIR_CANVAS)
    instance.start()
    instance.controller.mouse_press((0.3, 0.5), 0)
    instance.controller.mouse_drag((0.4, 0.5))
    instance.shutdown()
    instance.shutdown()
    assert source.stopped == 1
    assert not instance.controller.canvas.is_drawing
    assert instance._poll_job is None


def click_toolbar(instance, command):
    item = next(i for i in instance.controller.toolbar if i.command == command)
    x, y = instance._viewport.to_display((item.rect[0] + item.rect[2]) / 2, (item.rect[1] + item.rect[3]) / 2)
    instance._preview.event_generate("<ButtonPress-1>", x=round(x), y=round(y))
    instance._preview.event_generate("<ButtonRelease-1>", x=round(x), y=round(y))


def test_mouse_controls_have_real_behavior(app):  # noqa: F811
    """Toolbar items (colors, eraser, clear) are clicked with real tk mouse events
    on the camera toolbar; size and save come from the inspector."""
    from core.canvas import Tool
    from ui.app import Mode

    instance, _ = app
    instance.set_mode(Mode.AIR_CANVAS)
    pump(instance, 0.2)
    panel = instance._canvas_panel
    click_toolbar(instance, Command(Action.COLOR, "Blue"))
    assert instance.controller.color_name == "Blue"
    panel.size_buttons._command("Thin")  # The segmented button's own callback.
    assert instance.controller.size_name == "Thin"
    click_toolbar(instance, Command(Action.ERASER))
    assert instance.controller.canvas.tool is Tool.ERASER
    instance.controller.mouse_press((0.4, 0.5), 0)
    instance.controller.mouse_release()
    click_toolbar(instance, Command(Action.CLEAR))
    assert not instance.controller.canvas.is_empty  # First click only arms.
    assert instance.controller.clear_armed(time.monotonic())
    click_toolbar(instance, Command(Action.CLEAR))
    assert instance.controller.canvas.is_empty
    click_toolbar(instance, Command(Action.UNDO))
    assert not instance.controller.canvas.is_empty  # Clear is undoable.


def test_keyboard_shortcuts(app):  # noqa: F811
    from core.canvas import Tool
    from ui.app import Mode

    # Tk only delivers key events to a window that has OS focus, and the test
    # window deliberately never takes focus (so it can't interrupt the user).
    # So: verify every shortcut is bound on the window, then run the same
    # handler the binding calls. Real key presses need a manual check.
    instance, _ = app
    for sequence in [*instance.keymap, "<F1>", "<F2>"]:
        assert instance.bind(sequence), f"{sequence} is not bound"

    instance.set_mode(Mode.AIR_CANVAS)
    for key, check in [
        ("<Key-4>", lambda: instance.controller.color_name == "Mint"),
        ("<Key-e>", lambda: instance.controller.size_name == "Thick"),
        ("<Key-x>", lambda: instance.controller.canvas.tool is Tool.ERASER),
        ("<Key-q>", lambda: instance.controller.size_name == "Thin"),
    ]:
        instance._on_key_command(instance.keymap[key])
        assert check(), key
    instance.set_mode(Mode.HAND_TRACKING)
    instance._on_key_command(instance.keymap["<Key-1>"])  # Ignored outside Air Canvas.
    assert instance.controller.color_name == "Mint"
