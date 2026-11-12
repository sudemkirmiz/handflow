"""HandFlow Studio layout, states and rendering, in the real (invisible,
click-through, non-activating) window. Synthetic data only."""

import time

import numpy as np
import pytest

import config
from core.air_canvas import Action, Command
from core.camera import CameraState
from core.processing import FrameSnapshot, TrackingState
from tests.test_app import app, pump, tolerant_stale_limit  # noqa: F401  (pytest fixtures)

pytest.importorskip("customtkinter")


def walk(widget):
    for child in widget.winfo_children():
        yield child
        yield from walk(child)


def texts(widget):
    out = []
    for w in walk(widget):
        try:
            out.append(w.cget("text"))
        except Exception:
            pass
    return out


def _text(w):
    try:
        return w.cget("text")
    except Exception:
        return None


def box(w):
    return w.winfo_rootx(), w.winfo_rooty(), w.winfo_rootx() + w.winfo_width(), w.winfo_rooty() + w.winfo_height()


def inside(inner, outer, tol=1):
    return inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol and inner[2] <= outer[2] + tol \
        and inner[3] <= outer[3] + tol


def status(state, message, tracking=TrackingState.WAITING_FOR_CAMERA, frame=False, detail=""):
    return FrameSnapshot(1, time.monotonic(), state, message, tracking,
                         np.zeros((480, 640, 3), np.uint8) if frame else None, None, None, None, detail)


# --- palette and typography ---------------------------------------------------------------------

def test_palette_matches_the_design_spec():
    spec = {"background": "#111318", "surface": "#1B1F27", "elevated": "#292F3B", "accent": "#8B8CF8",
            "text": "#F4F5F8", "text_secondary": "#9CA3AF", "success": "#6EE7B7", "warning": "#F3C77B",
            "error": "#F28B93"}
    for key, value in spec.items():
        assert config.COLORS[key] == value
    assert not {"pink", "lilac", "card"} & set(config.COLORS)  # Old palette is gone.


def test_fonts_resolve_to_installed_families(app):  # noqa: F811
    import tkinter.font as tkfont

    from ui import theme

    instance, _ = app
    installed = set(tkfont.families(instance))
    for family in (theme.ui_family(instance), theme.semibold_family(instance), theme.mono_family(instance)):
        assert family in installed


# --- layout -------------------------------------------------------------------------------------

@pytest.mark.parametrize("geometry", ["960x600", "1180x720", "1600x900"])
@pytest.mark.parametrize("mode_name", ["HAND_TRACKING", "AIR_CANVAS"])
def test_layout_has_no_clipping_or_overlap(app, geometry, mode_name):  # noqa: F811
    from ui.app import Mode

    instance, _ = app
    instance.geometry(geometry)
    instance.set_mode(Mode[mode_name])
    instance.run_command(Command(Action.CLEAR))  # Puts a message in the Air Canvas inspector.
    pump(instance, 0.3)
    panel = instance._canvas_panel if mode_name == "AIR_CANVAS" else instance._analysis_card
    outer = box(panel)
    for w in walk(panel):
        if w.winfo_ismapped() and w.winfo_width() > 1:
            assert inside(box(w), outer), (geometry, w)
    # Header: brand, mode switch and camera status never overlap.
    switch = next(iter(instance._mode_buttons.values())).master
    brand = next(w for w in walk(instance) if getattr(w, "cget", None) and _text(w) == "HandFlow Studio")
    assert box(brand)[2] < box(switch)[0]  # Brand | mode switch
    assert box(switch)[2] < box(instance._camera_dot)[0]  # mode switch | camera status
    assert box(instance._camera_label)[2] <= box(instance)[2]  # Status not cut off on the right.
    # Inspector keeps its width; the workspace takes the rest.
    scaling = instance._get_window_scaling()
    assert abs(instance._sidebar.winfo_width() - 288 * scaling) <= 2
    assert instance._preview.winfo_width() > instance._sidebar.winfo_width()


def test_header_and_hint_follow_the_mode(app):  # noqa: F811
    from core.air_canvas import HINT
    from ui.app import HAND_TRACKING_HINT, Mode

    instance, _ = app
    assert "HandFlow Studio" in texts(instance)
    assert instance._hint_label.cget("text") == HAND_TRACKING_HINT
    instance.set_mode(Mode.AIR_CANVAS)
    assert instance._hint_label.cget("text") == HINT
    active = instance._mode_buttons[Mode.AIR_CANVAS].cget("text_color")
    inactive = instance._mode_buttons[Mode.HAND_TRACKING].cget("text_color")
    assert active == config.COLORS["text"] and inactive == config.COLORS["text_secondary"]


def test_inspector_does_not_duplicate_toolbar_buttons(app):  # noqa: F811
    from ui.app import Mode

    instance, _ = app
    instance.set_mode(Mode.AIR_CANVAS)
    labels = texts(instance._canvas_panel)
    for duplicate in ("Eraser", "Undo", "Clear"):
        assert duplicate not in labels  # These live on the virtual toolbar (gesture + mouse).
    assert "Save PNG" in labels and "Export Background" in labels
    assert {"Thin", "Medium", "Thick"} <= set(instance._canvas_panel.size_buttons._buttons_dict)


def test_save_button_is_disabled_until_there_is_a_drawing(app):  # noqa: F811
    from ui.app import Mode

    instance, _ = app
    instance.set_mode(Mode.AIR_CANVAS)
    pump(instance, 0.05)
    assert instance._canvas_panel.save_button.cget("state") == "disabled"
    instance.controller.mouse_press((0.4, 0.5), 0)
    instance.controller.mouse_drag((0.6, 0.5))
    instance.controller.mouse_release()
    instance._after_canvas_change()
    assert instance._canvas_panel.save_button.cget("state") == "normal"


def test_diagnostics_are_hidden_by_default(app):  # noqa: F811
    from ui.app import Mode

    instance, _ = app
    instance.set_mode(Mode.AIR_CANVAS)
    pump(instance, 0.1)
    assert instance.diagnostics is False
    assert instance._canvas_panel.debug_switch.get() == 0
    assert instance._preview.find_withtag("diag") == ()


# --- status and error states ------------------------------------------------------------------------

@pytest.mark.parametrize("state,message,header,color", [
    (CameraState.UNAVAILABLE, "Camera unavailable — test", "Camera unavailable", "error"),
    (CameraState.READ_FAILED, "Camera stopped", "Camera lost · reconnecting", "error"),
    (CameraState.DISCONNECTED, "Connecting to camera…", "Connecting…", "warning"),
])
def test_camera_states_in_header_and_preview(app, state, message, header, color):  # noqa: F811
    instance, source = app
    source.snapshot = status(state, message)
    instance.process_new_snapshots()
    assert instance._camera_label.cget("text") == header
    assert instance._camera_dot.cget("text_color") == config.COLORS[color]
    assert instance._preview.itemcget(instance._placeholder, "text") == message


def test_model_unavailable_is_shown_as_error(app):  # noqa: F811
    instance, source = app
    source.snapshot = status(CameraState.CONNECTED, "Connected · 640×480", TrackingState.MODEL_UNAVAILABLE,
                             frame=True, detail="Hand Landmarker model not found")
    instance.process_new_snapshots()
    assert instance._tracking_label.cget("text") == "Model unavailable"
    assert instance._tracking_label.cget("text_color") == config.COLORS["error"]
    assert "model not found" in instance._detail_label.cget("text")


# --- preview rendering and cleanup -------------------------------------------------------------------

def test_preview_reuses_its_image_and_cleans_up(app):  # noqa: F811
    from ui.app import Mode

    instance, source = app
    pump(instance, 0.2)
    for seq in range(1, 6):
        source.snapshot = FrameSnapshot(seq, time.monotonic(), CameraState.CONNECTED, "Connected",
                                        TrackingState.SEARCHING, np.full((480, 640, 3), seq, np.uint8),
                                        None, None, 30.0)
        instance.process_new_snapshots()
    photo = instance._photo
    assert instance.frames_rendered >= 5 and photo is not None
    source.snapshot = FrameSnapshot(6, time.monotonic(), CameraState.CONNECTED, "Connected",
                                    TrackingState.SEARCHING, np.zeros((480, 640, 3), np.uint8), None, None, 30.0)
    instance.process_new_snapshots()
    assert instance._photo is photo  # Same Tk image, pasted into: no per-frame allocation.
    vp = instance._viewport
    image_box = instance._preview.coords(instance._image_item)
    assert image_box == [vp.offset_x, vp.offset_y]  # Letterboxed, centered.
    assert abs(vp.display_w / vp.display_h - 640 / 480) < 0.01  # Aspect ratio preserved.
    instance.set_mode(Mode.AIR_CANVAS)
    instance.set_mode(Mode.HAND_TRACKING)
    assert instance._preview.find_withtag("overlay") == ()  # Overlays removed with the mode.


def test_fps_shown_in_workspace_footer(app):  # noqa: F811
    instance, source = app
    source.snapshot = FrameSnapshot(1, time.monotonic(), CameraState.CONNECTED, "Connected",
                                    TrackingState.SEARCHING, np.zeros((480, 640, 3), np.uint8), None, None, 29.7)
    instance.process_new_snapshots()
    assert instance._fps_value.cget("text") == "29.7 FPS"


# --- virtual toolbar states ---------------------------------------------------------------------------

def item_fill_outline(instance, index):
    cv = instance._preview
    item = cv.find_withtag(f"tb{index}")[0]
    return cv.itemcget(item, "fill"), cv.itemcget(item, "outline")


def test_toolbar_states_use_accent_and_warning(app):  # noqa: F811
    from ui.app import Mode

    instance, _ = app
    instance.set_mode(Mode.AIR_CANVAS)
    pump(instance, 0.2)
    ctl = instance.controller
    indexes = {i.command: n for n, i in enumerate(ctl.toolbar)}
    pink = indexes[Command(Action.COLOR, "Pink")]
    blue = indexes[Command(Action.COLOR, "Blue")]
    assert item_fill_outline(instance, pink)[1] == config.COLORS["accent"]  # Selected color.
    assert item_fill_outline(instance, blue)[1] != config.COLORS["accent"]
    ctl.mouse_press((0.4, 0.5), 0)
    ctl.mouse_release()
    instance.run_command(Command(Action.CLEAR))  # Arms Clear.
    clear = indexes[Command(Action.CLEAR)]
    assert item_fill_outline(instance, clear)[1] == config.COLORS["warning"]
    labels = [instance._preview.itemcget(i, "text") for i in instance._preview.find_withtag("toolbar")
              if instance._preview.type(i) == "text"]
    assert "Confirm" in labels
