"""One-finger interaction: the raised index finger draws on the canvas and
selects tools on the toolbar. Synthetic input through the real finger counter;
this does not replace testing with a physical hand."""

import math
import random
from pathlib import Path

import pytest

import config
from core.air_canvas import (
    DRAW_MIN_Y, DRAW_START_Y, TOOLBAR_ZONE_BOTTOM, Action, AirCanvasController, Command, Zone, zone_of,
)
from core.canvas import DrawingCanvas, Tool
from core.gestures import Gesture
from tests.gesture_helpers import DRAW_FINGERS, FIST_FINGERS, UNCERTAIN_INDEX, Feeder

POINT = DRAW_FINGERS  # Index finger only.
NOISE = UNCERTAIN_INDEX


class Spy:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return Path(f"spy_{self.calls}.png")


@pytest.fixture
def ctl():
    return AirCanvasController(DrawingCanvas(), save_handler=Spy())


def item_for(ctl, action, value=""):
    return next(i for i in ctl.toolbar if i.command == Command(action, value))


def center(item):
    x0, y0, x1, y1 = item.rect
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def glide(start, end, n):
    """Finger moving from start to end over n frames (realistic, no teleporting)."""
    return [(start[0] + (end[0] - start[0]) * k / n, start[1] + (end[1] - start[1]) * k / n) for k in range(1, n + 1)]


def all_points(ctl):
    pts = [p for s in ctl.canvas.strokes for p in s.points]
    if ctl.canvas.active_stroke:
        pts += list(ctl.canvas.active_stroke.points)
    return pts


# --- regions -------------------------------------------------------------------------------------

def test_regions_are_ordered_and_explicit():
    assert TOOLBAR_ZONE_BOTTOM < DRAW_MIN_Y < DRAW_START_Y
    assert zone_of((0.5, 0.08)) is Zone.TOOLBAR
    assert zone_of((0.5, TOOLBAR_ZONE_BOTTOM)) is Zone.TOOLBAR
    assert zone_of((0.5, (TOOLBAR_ZONE_BOTTOM + DRAW_START_Y) / 2)) is Zone.BUFFER
    assert zone_of((0.5, DRAW_START_Y)) is Zone.CANVAS
    assert zone_of((0.5, 0.9)) is Zone.CANVAS


def test_default_brush_size_is_medium(ctl):
    assert ctl.size_name == "Medium"
    assert ctl.canvas.thickness == config.BRUSH_SIZES["Medium"]


# --- drawing and selecting with the same finger -------------------------------------------------------

def test_index_finger_draws_inside_canvas(ctl):
    f = Feeder(ctl)
    f.path(POINT, glide((0.2, 0.5), (0.6, 0.6), 30))
    assert ctl.interaction == "Drawing"
    f.hold(FIST_FINGERS, (0.6, 0.6), 0.3)
    assert len(ctl.canvas.strokes) == 1 and len(ctl.canvas.strokes[0].points) > 10


@pytest.mark.parametrize("color", list(config.BRUSH_COLORS))
def test_same_finger_selects_every_color(ctl, color):
    f = Feeder(ctl)
    f.path(POINT, glide((0.5, 0.6), (0.5, 0.4), 6))  # Start in the canvas.
    target = center(item_for(ctl, Action.COLOR, color))
    f.path(POINT, glide((0.5, 0.4), target, 12))  # Move up into the toolbar.
    f.hold(POINT, target, config.DWELL_SECONDS + 0.3)
    assert ctl.color_name == color or color == "Pink"  # Pink is already the default.
    assert ctl.interaction == "Selecting"


def test_same_finger_selects_size_and_eraser(ctl):
    f = Feeder(ctl)
    size = center(item_for(ctl, Action.CYCLE_SIZE))
    eraser = center(item_for(ctl, Action.ERASER))
    f.path(POINT, glide((0.5, 0.5), size, 10))
    f.hold(POINT, size, config.DWELL_SECONDS + 0.3)
    assert ctl.size_name == "Thick"
    f.path(POINT, glide(size, eraser, 6))
    f.hold(POINT, eraser, config.DWELL_SECONDS + 0.3)
    assert ctl.canvas.tool is Tool.ERASER


def test_selected_color_applies_to_the_next_stroke(ctl):
    f = Feeder(ctl)
    mint = center(item_for(ctl, Action.COLOR, "Mint"))
    f.path(POINT, glide((0.4, 0.5), mint, 10))
    f.hold(POINT, mint, config.DWELL_SECONDS + 0.3)
    f.path(POINT, glide(mint, (0.4, 0.6), 12))  # Back down, still pointing.
    f.path(POINT, glide((0.4, 0.6), (0.7, 0.6), 20))
    f.hold(FIST_FINGERS, (0.7, 0.6), 0.3)
    assert ctl.canvas.strokes[-1].color == config.hex_to_rgb(config.BRUSH_COLORS["Mint"])


# --- toolbar never creates strokes ------------------------------------------------------------------

def test_toolbar_hover_never_creates_a_stroke(ctl):
    f = Feeder(ctl)
    rng = random.Random(3)
    for item in ctl.toolbar:  # Wander over every item with jitter and noisy frames.
        x, y = center(item)
        for i in range(20):
            f.frame(NOISE if i % 6 == 5 else POINT, (x + rng.gauss(0, 0.01), y + rng.gauss(0, 0.01)))
    assert ctl.canvas.strokes == () and ctl.canvas.active_stroke is None


def test_entering_toolbar_ends_stroke_and_return_starts_clean(ctl):
    f = Feeder(ctl)
    f.path(POINT, glide((0.3, 0.6), (0.3, 0.3), 15))
    f.path(POINT, glide((0.3, 0.3), (0.35, 0.08), 12))  # Up into the toolbar...
    assert not ctl.canvas.is_drawing
    f.path(POINT, glide((0.35, 0.08), (0.6, 0.08), 10))  # ...across it...
    f.path(POINT, glide((0.6, 0.08), (0.6, 0.6), 20))  # ...and back down.
    f.hold(FIST_FINGERS, (0.6, 0.6), 0.3)
    first, second = ctl.canvas.strokes
    assert all(p[1] >= DRAW_MIN_Y for p in all_points(ctl))  # Nothing near the toolbar.
    assert second.points[0][1] >= DRAW_START_Y  # Clean new stroke below the buffer.
    assert math.dist(first.points[-1], second.points[0]) > 0.2  # No connecting line.


def test_jitter_at_canvas_edge_does_not_chop_the_stroke(ctl):
    f = Feeder(ctl)
    rng = random.Random(5)
    edge_y = (DRAW_MIN_Y + DRAW_START_Y) / 2  # Between "may continue" and "may start".
    f.path(POINT, glide((0.2, 0.45), (0.3, DRAW_START_Y + 0.01), 12))
    for k in range(60):  # Drawing along the top edge of the canvas, jittering.
        f.frame(POINT, (0.3 + 0.006 * k, edge_y + rng.gauss(0, 0.003)))
    f.hold(FIST_FINGERS, (0.66, 0.3), 0.3)
    assert len(ctl.canvas.strokes) <= 2
    assert sum(len(s.points) for s in ctl.canvas.strokes) > 40  # It really drew along the edge.


def test_buffer_zone_neither_draws_nor_selects(ctl):
    f = Feeder(ctl)
    y = (TOOLBAR_ZONE_BOTTOM + DRAW_START_Y) / 2
    f.path(POINT, glide((0.2, y), (0.8, y), 60))
    assert ctl.canvas.is_empty and ctl.executed == []


# --- accidental activation -----------------------------------------------------------------------

def test_sweeping_across_toolbar_selects_nothing(ctl):
    f = Feeder(ctl)
    y = center(ctl.toolbar[0])[1]
    for _ in range(3):  # Back and forth at ~0.9 frame widths per second.
        f.path(POINT, glide((0.04, y), (0.96, y), 30))
        f.path(POINT, glide((0.96, y), (0.04, y), 30))
    assert ctl.executed == []


def test_uncertain_frames_alone_never_select(ctl):
    f = Feeder(ctl)
    target = center(item_for(ctl, Action.COLOR, "Blue"))
    f.hold(NOISE, target, 3.0)
    assert ctl.executed == [] and ctl.gesture is Gesture.UNKNOWN


def test_brief_uncertainty_does_not_add_dwell_time(ctl):
    f = Feeder(ctl)
    target = center(item_for(ctl, Action.COLOR, "Blue"))
    f.hold(POINT, target, 0.2)
    for _ in range(4):  # Alternate: 2 clean frames, 6 uncertain frames.
        f.hold(POINT, target, 2 / 30)
        f.hold(NOISE, target, 6 / 30)
    # About 0.27 s of clean hovering in total, less than the 0.5 s dwell.
    assert ctl.color_name == "Pink"


def test_fist_over_toolbar_selects_nothing(ctl):
    f = Feeder(ctl)
    f.hold(FIST_FINGERS, center(item_for(ctl, Action.COLOR, "Blue")), 3.0)
    assert ctl.executed == []


def test_real_hand_loss_cancels_hover_and_progress_restarts(ctl):
    f = Feeder(ctl)
    target = center(item_for(ctl, Action.COLOR, "Mint"))
    f.hold(POINT, target, 0.45)
    assert ctl.hovered is not None and ctl.dwell_progress > 0
    for _ in range(round((config.DWELL_GRACE_SECONDS + 0.1) * 30)):
        f.frame(None, None)  # Hand gone for longer than a drop-out.
    assert ctl.hovered is None and ctl.dwell_progress == 0.0
    f.hold(POINT, target, 0.2)  # Hand back: gesture must stabilize, timer starts at zero.
    assert ctl.color_name == "Pink"


def test_brief_tracking_dropout_only_pauses_hover(ctl):
    f = Feeder(ctl)
    target = center(item_for(ctl, Action.UNDO))
    f.hold(POINT, target, 0.6)
    progress = ctl.dwell_progress
    f.frame(None, None)  # One lost frame, as near the image edge.
    f.frame(None, None)
    assert ctl.hovered is not None
    assert ctl.dwell_progress == pytest.approx(progress)  # Paused: not reset, not advanced.
    f.hold(POINT, target, config.ACTION_DWELL_SECONDS)
    assert ctl.executed == [Command(Action.UNDO)]


def test_dropouts_lift_the_pen_and_never_bridge_a_gap(ctl):
    f = Feeder(ctl)
    f.path(POINT, glide((0.2, 0.6), (0.4, 0.6), 15))
    f.frame(None, None)
    assert not ctl.canvas.is_drawing  # No hand: pen up at once (drawing stays strict).
    f.path(POINT, glide((0.7, 0.8), (0.8, 0.8), 10))  # Back 33 ms later, elsewhere.
    f.hold(FIST_FINGERS, (0.8, 0.8), 0.3)
    first, second = ctl.canvas.strokes
    assert math.dist(first.points[-1], second.points[0]) > 0.3
    assert second.points[0] == pytest.approx((0.7 + 0.01, 0.8), abs=1e-9)  # Started right away.


def test_dropout_resume_needs_the_same_gesture(ctl):
    f = Feeder(ctl)
    f.path(POINT, glide((0.2, 0.6), (0.4, 0.6), 15))
    f.frame(None, None)
    f.frame(FIST_FINGERS, (0.4, 0.6))  # Different gesture after the drop-out.
    assert ctl.gesture is Gesture.NO_HAND  # Must be confirmed normally, not resumed.


def test_actions_need_the_longer_dwell(ctl):
    f = Feeder(ctl)
    undo = center(item_for(ctl, Action.UNDO))
    f.hold(POINT, undo, 0.15 + config.DWELL_SECONDS + 0.1)  # Enough for a color, not for Undo.
    assert ctl.executed == []
    f.hold(POINT, undo, config.ACTION_DWELL_SECONDS)
    assert ctl.executed == [Command(Action.UNDO)]


@pytest.mark.parametrize("action", [Action.UNDO, Action.SAVE, Action.CLEAR])
def test_held_finger_fires_undo_save_clear_once(ctl, action):
    for x in (0.3, 0.5, 0.7):
        ctl.canvas.begin_stroke((x, 0.6))
        ctl.canvas.add_point((x, 0.7))
        ctl.canvas.end_stroke()
    f = Feeder(ctl)
    f.hold(POINT, center(item_for(ctl, action)), 10.0)
    if action is Action.CLEAR:
        # Hold-to-confirm: arm + confirm = 2 commands, one actual clear, then latched.
        assert ctl.executed.count(Command(action)) == 2
        assert ctl.canvas.is_empty and len(ctl.canvas.strokes) == 0
        ctl.execute(Command(Action.UNDO), f.t)
        assert len(ctl.canvas.strokes) == 3  # Still undoable.
    else:
        assert ctl.executed.count(Command(action)) == 1
        assert len(ctl.canvas.strokes) == (2 if action is Action.UNDO else 3)
    assert ctl.save_handler.calls == (1 if action is Action.SAVE else 0)


def test_clear_with_one_finger_needs_confirmation(ctl):
    ctl.canvas.begin_stroke((0.5, 0.6))
    ctl.canvas.end_stroke()
    f = Feeder(ctl)
    clear = center(item_for(ctl, Action.CLEAR))
    f.hold(POINT, clear, 0.15 + config.ACTION_DWELL_SECONDS + 0.1)
    assert ctl.clear_armed(f.t) and not ctl.canvas.is_empty
    f.path(POINT, glide(clear, (clear[0], 0.5), 8))  # Leave...
    f.path(POINT, glide((clear[0], 0.5), clear, 8))  # ...and come back.
    f.hold(POINT, clear, config.COMMAND_COOLDOWN_SECONDS + config.SAVE_CLEAR_COOLDOWN_SECONDS
           + config.ACTION_DWELL_SECONDS)
    assert ctl.canvas.is_empty
    ctl.execute(Command(Action.UNDO), f.t + 1)
    assert not ctl.canvas.is_empty  # Clear stays undoable.


def test_clear_confirm_by_holding_never_touches_neighbours(ctl):
    ctl.canvas.begin_stroke((0.5, 0.6))
    ctl.canvas.end_stroke()
    f = Feeder(ctl)
    rng = random.Random(21)
    clear = center(item_for(ctl, Action.CLEAR))
    f.path(POINT, glide((0.5, 0.6), clear, 12))
    for _ in range(round((0.2 + 2 * config.ACTION_DWELL_SECONDS + 0.3) * 30)):
        f.frame(POINT, (clear[0] + rng.gauss(0, 0.004), clear[1] + rng.gauss(0, 0.004)))
    assert ctl.canvas.is_empty
    assert {c.action for c in ctl.executed} == {Action.CLEAR}  # No Undo, no Save.


def test_clear_armed_then_abandoned_stays_safe(ctl):
    ctl.canvas.begin_stroke((0.5, 0.6))
    ctl.canvas.end_stroke()
    f = Feeder(ctl)
    clear = center(item_for(ctl, Action.CLEAR))
    f.hold(POINT, clear, 0.15 + config.ACTION_DWELL_SECONDS + 0.1)
    assert ctl.clear_armed(f.t)
    f.path(POINT, glide(clear, (0.5, 0.6), 10))  # Walks away and draws instead.
    f.hold(POINT, (0.5, 0.6), config.CLEAR_CONFIRM_SECONDS + 0.5)
    assert not ctl.canvas.strokes == () and not ctl.clear_armed(f.t)


def test_jitter_on_border_does_not_select_neighbour(ctl):
    f = Feeder(ctl)
    blue, mint = item_for(ctl, Action.COLOR, "Blue"), item_for(ctl, Action.COLOR, "Mint")
    rng = random.Random(11)
    f.path(POINT, glide((0.4, 0.5), center(blue), 10))
    edge = (blue.rect[2] - 0.002, center(blue)[1])
    for _ in range(60):
        f.frame(POINT, (edge[0] + rng.gauss(0, 0.004), edge[1] + rng.gauss(0, 0.004)))
    assert ctl.color_name == "Blue"
    assert Command(Action.COLOR, "Mint") not in ctl.executed


# --- mouse still works -------------------------------------------------------------------------------

def test_mouse_click_on_toolbar_and_mouse_drawing(ctl):
    ctl.mouse_press(center(item_for(ctl, Action.COLOR, "Yellow")), now=1.0)
    ctl.mouse_release()
    ctl.mouse_press((0.3, 0.6), now=2.0)
    ctl.mouse_drag((0.6, 0.6))
    ctl.mouse_release()
    assert ctl.color_name == "Yellow" and len(ctl.canvas.strokes) == 1
