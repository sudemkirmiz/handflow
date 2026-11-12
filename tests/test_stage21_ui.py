"""Stage 2.1 UI checks. The window is invisible and never takes focus (see the
``app`` fixture), so these tests do not interrupt the desktop."""

import time

import numpy as np
import pytest

from core.air_canvas import Action, Command
from core.camera import CameraState
from core.finger_counter import Finger
from core.hand_tracker import Landmark, TrackedHand
from core.processing import FrameSnapshot, TrackingState
from tests.gesture_helpers import SELECT_FINGERS
from tests.synthetic_hand import make_hand
from tests.test_app import app, pump, tolerant_stale_limit  # noqa: F401  (pytest fixtures)

pytest.importorskip("customtkinter")


def snapshot(seq, u, v, frame_shape=(480, 640), fingers=SELECT_FINGERS, t=None):
    world = make_hand({Finger.INDEX, Finger.MIDDLE})
    image = [Landmark(0.5, 0.5, 0.0)] * 21
    image[8] = Landmark(u, v, 0.0)
    hand = TrackedHand(tuple(image), tuple(Landmark(*p) for p in world), "Right", 0.9)
    return FrameSnapshot(
        sequence=seq, created_at=t if t is not None else time.monotonic(),
        camera_state=CameraState.CONNECTED, camera_message="Connected",
        tracking_state=TrackingState.TRACKING,
        frame_rgb=np.zeros((*frame_shape, 3), np.uint8), hand=hand, fingers=fingers, fps=30.0,
    )


def bbox_center(cv, item_id):
    x0, y0, x1, y1 = cv.coords(item_id)
    return (x0 + x1) / 2, (y0 + y1) / 2


def contains(bbox, x, y):
    x0, y0, x1, y1 = bbox
    return x0 <= x <= x1 and y0 <= y <= y1


# --- 9. pointer mapping at different viewport sizes / camera resolutions ------------------------------

@pytest.mark.parametrize("geometry", ["960x600", "1180x720", "1500x640", "1000x860"])
@pytest.mark.parametrize("frame_shape", [(480, 640), (720, 1280)])
def test_cursor_and_toolbar_share_one_mapping(app, geometry, frame_shape):  # noqa: F811
    from ui.app import Mode

    instance, source = app
    instance.geometry(geometry)
    instance.set_mode(Mode.AIR_CANVAS)
    pump(instance, 0.25)
    ctl, cv = instance.controller, instance._preview
    seq = 0
    for index in (0, 5, 8, 10):  # Pink, White, Undo, Save.
        x0, y0, x1, y1 = ctl.toolbar[index].rect
        u, v = (x0 + x1) / 2, (y0 + y1) / 2
        t0 = time.monotonic() - 0.5
        for k in range(12):  # Settle the filter on the item.
            seq += 1
            source.snapshot = snapshot(seq, u, v, frame_shape, t=t0 + k / 30)
        instance.process_new_snapshots()
        vp = instance._viewport
        assert (vp.frame_w, vp.frame_h) == (frame_shape[1], frame_shape[0])
        expected = vp.to_display(*ctl.pointer)
        ring = instance._cursor_items[0]
        cx, cy = bbox_center(cv, ring)
        assert (cx, cy) == pytest.approx(expected, abs=1.0)  # Cursor drawn where the pointer maps.
        # The drawn toolbar item is under the cursor, and it is the item being hovered.
        assert contains(cv.bbox(f"tb{index}"), cx, cy)
        assert ctl.hovered is ctl.toolbar[index]
        # Inverse mapping (used by the mouse) lands on the same item.
        back = vp.to_normalized(cx, cy)
        assert ctl.toolbar[index].contains(back)


def test_mouse_clicks_on_every_toolbar_item(app):  # noqa: F811
    from ui.app import Mode

    instance, _ = app
    instance.set_mode(Mode.AIR_CANVAS)
    pump(instance, 0.25)
    ctl, vp, cv = instance.controller, instance._viewport, instance._preview
    ctl.mouse_press((0.4, 0.5), 0)
    ctl.mouse_drag((0.6, 0.5))
    ctl.mouse_release()
    for item in ctl.toolbar:
        x0, y0, x1, y1 = item.rect
        px, py = vp.to_display((x0 + x1) / 2, (y0 + y1) / 2)
        before = len(ctl.executed)
        cv.event_generate("<ButtonPress-1>", x=round(px), y=round(py))
        cv.event_generate("<ButtonRelease-1>", x=round(px), y=round(py))
        assert len(ctl.executed) == before + 1, item.label
    actions = {c.action for c in ctl.executed}
    assert actions == {Action.COLOR, Action.SIZE, Action.ERASER, Action.UNDO, Action.CLEAR, Action.SAVE}


# --- diagnostics ------------------------------------------------------------------------------------

def test_diagnostics_are_off_by_default_and_toggle_cleanly(app):  # noqa: F811
    from ui.app import Mode

    instance, source = app
    instance.set_mode(Mode.AIR_CANVAS)
    t0 = time.monotonic() - 0.3
    for k in range(1, 9):
        source.snapshot = snapshot(k, 0.5, 0.6, t=t0 + k / 30)
    instance.process_new_snapshots()
    pump(instance, 0.1)
    cv = instance._preview
    assert cv.find_withtag("diag") == ()  # Normal UI is uncluttered.

    instance.set_diagnostics(True)
    assert instance._canvas_panel.debug_switch.get() == 1
    text = cv.itemcget(instance._diag_items[1], "text")
    for needle in ("raw: Select", "stable: Select", "tip raw: 0.500, 0.600", "filtered:", "mapped:", "hover:",
                   "dwell:", "status:"):
        assert needle in text, needle

    instance.set_diagnostics(False)
    assert cv.find_withtag("diag") == ()
    assert instance._canvas_panel.debug_switch.get() == 0


def test_diagnostics_show_rejection_reason(app):  # noqa: F811
    from tests.gesture_helpers import UNCERTAIN_INDEX
    from ui.app import Mode

    instance, source = app
    instance.set_mode(Mode.AIR_CANVAS)
    instance.set_diagnostics(True)
    source.snapshot = snapshot(1, 0.5, 0.6, fingers=UNCERTAIN_INDEX)
    instance.process_new_snapshots()
    text = instance.diagnostics_text()
    assert "raw: Unknown (uncertain: Index)" in text
    assert "top reject: uncertain: Index" in text


def test_debug_switch_toggles_diagnostics(app):  # noqa: F811
    from ui.app import Mode

    instance, _ = app
    instance.set_mode(Mode.AIR_CANVAS)
    instance._canvas_panel.debug_switch.toggle()
    assert instance.diagnostics is True
    instance._canvas_panel.debug_switch.toggle()
    assert instance.diagnostics is False


# --- layout -----------------------------------------------------------------------------------------

def test_air_canvas_panel_fits_at_minimum_window_size(app):  # noqa: F811
    from ui.app import Mode

    instance, _ = app
    instance.geometry("960x600")
    instance.set_mode(Mode.AIR_CANVAS)
    instance.run_command(Command(Action.CLEAR))  # Longest messages are two lines.
    pump(instance, 0.3)
    panel = instance._canvas_panel
    bottom = panel.message_label.winfo_y() + panel.message_label.winfo_height()
    assert bottom <= panel.winfo_height()
    top_row = panel.debug_switch.master
    assert panel.debug_switch.winfo_x() + panel.debug_switch.winfo_width() <= panel.tracking_label.winfo_x() + 2
    assert top_row.winfo_width() <= panel.winfo_width()
