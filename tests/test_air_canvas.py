import math
from pathlib import Path

import pytest

import config
from core.air_canvas import DRAW_MIN_Y, Action, AirCanvasController, Command
from core.canvas import DrawingCanvas, Tool
from core.gestures import Gesture
from tests.gesture_helpers import (
    DRAW_FINGERS, FIST_FINGERS, SELECT_FINGERS, UNCERTAIN_INDEX, Feeder,
)


class SaveSpy:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return Path(f"handflow_test_{self.calls}.png")


@pytest.fixture
def ctl():
    return AirCanvasController(DrawingCanvas(), save_handler=SaveSpy())


def item_center(ctl, action, value=""):
    item = next(i for i in ctl.toolbar if i.command == Command(action, value))
    x0, y0, x1, y1 = item.rect
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def line(start, end, n=10):
    return [(start[0] + (end[0] - start[0]) * k / n, start[1] + (end[1] - start[1]) * k / n) for k in range(n + 1)]


# --- drawing ----------------------------------------------------------------------

def test_index_finger_draws_and_fist_lifts_pen(ctl):
    f = Feeder(ctl)
    f.path(DRAW_FINGERS, line((0.2, 0.5), (0.6, 0.5), 20))
    assert ctl.gesture is Gesture.DRAW
    assert ctl.canvas.is_drawing
    f.hold(FIST_FINGERS, (0.6, 0.5), 0.3)
    assert ctl.gesture is Gesture.IDLE
    assert not ctl.canvas.is_drawing
    assert len(ctl.canvas.strokes) == 1
    stroke = ctl.canvas.strokes[0]
    assert len(stroke.points) > 5
    assert stroke.points[-1][0] == pytest.approx(0.6, abs=0.03)


def test_no_points_before_gesture_is_stable(ctl):
    f = Feeder(ctl)
    f.frame(DRAW_FINGERS, (0.3, 0.5))
    f.frame(DRAW_FINGERS, (0.3, 0.5))
    assert not ctl.canvas.is_drawing  # 2 frames < hold time.


def test_uncertain_gesture_never_draws(ctl):
    Feeder(ctl).path(UNCERTAIN_INDEX, line((0.2, 0.5), (0.8, 0.5), 30))
    assert ctl.gesture is Gesture.UNKNOWN
    assert ctl.canvas.is_empty


def test_no_drawing_under_toolbar(ctl):
    Feeder(ctl).path(DRAW_FINGERS, line((0.2, 0.08), (0.8, 0.08), 30))
    assert ctl.canvas.is_empty


# --- 4. DRAW -> SELECT ----------------------------------------------------------------

def test_draw_to_select_ends_stroke_without_extra_points(ctl):
    f = Feeder(ctl)
    f.path(DRAW_FINGERS, line((0.2, 0.5), (0.4, 0.5), 15))
    points_before = len(ctl.canvas.active_stroke.points)
    # Switch to SELECT while moving: none of these positions may be drawn.
    f.path(SELECT_FINGERS, line((0.4, 0.5), (0.4, 0.9), 15))
    assert ctl.gesture is Gesture.SELECT
    assert not ctl.canvas.is_drawing
    assert len(ctl.canvas.strokes) == 1
    assert len(ctl.canvas.strokes[0].points) == points_before
    assert max(p[1] for p in ctl.canvas.strokes[0].points) < 0.51


def test_pointing_over_toolbar_never_draws(ctl):
    # One-finger model: the drawing gesture may select tools, but never draws there.
    f = Feeder(ctl)
    f.hold(DRAW_FINGERS, item_center(ctl, Action.SAVE), 3.0)
    f.hold(DRAW_FINGERS, item_center(ctl, Action.CLEAR), 3.0)
    assert ctl.canvas.is_empty
    assert ctl.executed == [Command(Action.SAVE), Command(Action.CLEAR)]  # Each once.
    assert ctl.save_handler.calls == 0  # Nothing to save yet.


# --- 5. tracking loss during a stroke -------------------------------------------------------

def test_hand_loss_commits_stroke_and_recovery_starts_new_one(ctl):
    f = Feeder(ctl)
    f.path(DRAW_FINGERS, line((0.2, 0.5), (0.3, 0.5), 10))
    f.frame(None, None)  # Hand lost.
    assert not ctl.canvas.is_drawing
    assert len(ctl.canvas.strokes) == 1
    f.path(DRAW_FINGERS, line((0.8, 0.8), (0.85, 0.8), 10))  # Reappears far away.
    f.hold(FIST_FINGERS, (0.85, 0.8), 0.3)
    assert len(ctl.canvas.strokes) == 2
    first, second = ctl.canvas.strokes
    assert max(p[0] for p in first.points) < 0.31  # No line across the gap.
    assert min(p[0] for p in second.points) > 0.79


def test_time_gap_splits_stroke_even_at_same_position(ctl):
    f = Feeder(ctl)
    f.path(DRAW_FINGERS, line((0.2, 0.5), (0.3, 0.5), 10))
    f.frame(DRAW_FINGERS, (0.32, 0.5), dt=1.0)  # Camera stalled for a second.
    f.frame(DRAW_FINGERS, (0.34, 0.5))
    assert len(ctl.canvas.strokes) == 1
    assert ctl.canvas.is_drawing


def test_large_jump_splits_stroke(ctl):
    f = Feeder(ctl)
    f.path(DRAW_FINGERS, line((0.2, 0.5), (0.3, 0.5), 10))
    for _ in range(4):
        f.frame(DRAW_FINGERS, (0.9, 0.9))  # Tracking glitch / re-detection elsewhere.
    assert len(ctl.canvas.strokes) == 1
    assert all(math.dist(a, b) < 0.15 for a, b in zip(ctl.canvas.strokes[0].points, ctl.canvas.strokes[0].points[1:]))


def test_stale_tracking_lifts_pen(ctl):
    f = Feeder(ctl)
    f.path(DRAW_FINGERS, line((0.2, 0.5), (0.3, 0.5), 10))
    ctl.tracking_lost()
    assert not ctl.canvas.is_drawing and len(ctl.canvas.strokes) == 1
    assert ctl.gesture is Gesture.NO_HAND


# --- 3. dwell timing / 12. cooldown -----------------------------------------------------

def test_dwell_requires_full_interval(ctl):
    f = Feeder(ctl)
    target = item_center(ctl, Action.COLOR, "Mint")
    f.hold(SELECT_FINGERS, target, 0.25)  # Becomes stable SELECT.
    f.hold(SELECT_FINGERS, target, config.DWELL_SECONDS * 0.5)
    assert ctl.hovered is not None and 0 < ctl.dwell_progress < 1
    assert ctl.color_name == "Pink"
    f.hold(SELECT_FINGERS, target, config.DWELL_SECONDS * 0.6)
    assert ctl.color_name == "Mint"


def test_moving_off_item_resets_dwell(ctl):
    f = Feeder(ctl)
    a, b = item_center(ctl, Action.COLOR, "Blue"), item_center(ctl, Action.COLOR, "Mint")
    for _ in range(6):  # Alternate faster than the dwell time.
        f.hold(SELECT_FINGERS, a, config.DWELL_SECONDS * 0.6)
        f.hold(SELECT_FINGERS, b, config.DWELL_SECONDS * 0.6)
    assert ctl.executed == []


def test_held_save_gesture_saves_once(ctl):
    f = Feeder(ctl)
    f.hold(DRAW_FINGERS, (0.5, 0.5), 0.5)  # Something to save.
    f.hold(SELECT_FINGERS, item_center(ctl, Action.SAVE), 10.0)
    assert ctl.save_handler.calls == 1
    assert ctl.message.startswith("Saved")


def test_reentering_item_runs_again_only_after_cooldown(ctl):
    f = Feeder(ctl)
    undo = item_center(ctl, Action.UNDO)
    f.hold(SELECT_FINGERS, undo, 0.25 + config.ACTION_DWELL_SECONDS + 0.1)
    assert ctl.executed == [Command(Action.UNDO)]
    f.frame(SELECT_FINGERS, (0.5, 0.6))  # Leave the item (re-arms the latch) ...
    # ... and come back: the cooldown holds the dwell timer, then a full dwell is needed.
    f.hold(SELECT_FINGERS, undo, config.ACTION_DWELL_SECONDS * 0.9)
    assert len(ctl.executed) == 1
    f.hold(SELECT_FINGERS, undo, config.COMMAND_COOLDOWN_SECONDS + 0.3)
    assert len(ctl.executed) == 2


def test_cooldown_delays_next_command(ctl):
    f = Feeder(ctl)
    f.hold(SELECT_FINGERS, item_center(ctl, Action.COLOR, "Blue"), 0.25 + config.DWELL_SECONDS + 0.05)
    assert ctl.color_name == "Blue"
    # Immediately dwell on another color: cooldown must hold progress at zero first.
    f.hold(SELECT_FINGERS, item_center(ctl, Action.COLOR, "Mint"), config.DWELL_SECONDS * 0.9)
    assert ctl.color_name == "Blue"
    f.hold(SELECT_FINGERS, item_center(ctl, Action.COLOR, "Mint"), config.COMMAND_COOLDOWN_SECONDS + 0.2)
    assert ctl.color_name == "Mint"


def test_duplicate_snapshots_are_ignored(ctl):
    assert ctl.process(5, 1.0, DRAW_FINGERS, (0.5, 0.5)) is True
    assert ctl.process(5, 1.0, DRAW_FINGERS, (0.5, 0.5)) is False
    assert ctl.process(4, 1.1, DRAW_FINGERS, (0.5, 0.5)) is False


# --- 8. color / size --------------------------------------------------------------------

def test_color_and_size_apply_to_new_strokes_only(ctl):
    f = Feeder(ctl)
    f.path(DRAW_FINGERS, line((0.2, 0.5), (0.3, 0.5)))
    ctl.execute(Command(Action.COLOR, "Yellow"), 0)
    ctl.execute(Command(Action.SIZE, "Thick"), 0)
    f.hold(FIST_FINGERS, (0.3, 0.5), 0.3)
    f.path(DRAW_FINGERS, line((0.2, 0.7), (0.3, 0.7)))
    f.hold(FIST_FINGERS, (0.3, 0.7), 0.3)
    first, second = ctl.canvas.strokes
    assert first.color == config.hex_to_rgb(config.BRUSH_COLORS["Pink"])
    assert first.thickness == config.BRUSH_SIZES["Medium"]
    assert second.color == config.hex_to_rgb(config.BRUSH_COLORS["Yellow"])
    assert second.thickness == config.BRUSH_SIZES["Thick"]


def test_size_cycle(ctl):
    names = []
    for _ in range(3):
        ctl.execute(Command(Action.CYCLE_SIZE), 0)
        names.append(ctl.size_name)
    assert names == ["Thick", "Thin", "Medium"]


def test_eraser_toggle_and_color_returns_to_brush(ctl):
    ctl.execute(Command(Action.ERASER), 0)
    assert ctl.canvas.tool is Tool.ERASER
    ctl.execute(Command(Action.COLOR, "Blue"), 0)
    assert ctl.canvas.tool is Tool.BRUSH


# --- 10. clear safety ---------------------------------------------------------------------

def test_clear_needs_confirmation_and_is_undoable(ctl):
    f = Feeder(ctl)
    f.hold(DRAW_FINGERS, (0.5, 0.5), 0.5)
    f.hold(FIST_FINGERS, (0.5, 0.5), 0.3)
    msg = ctl.execute(Command(Action.CLEAR), 10.0)
    assert "again" in msg and len(ctl.canvas.strokes) == 1 and ctl.clear_armed(10.0)
    ctl.execute(Command(Action.CLEAR), 11.0)
    assert ctl.canvas.is_empty
    ctl.execute(Command(Action.UNDO), 12.0)
    assert len(ctl.canvas.strokes) == 1


def test_clear_confirmation_expires(ctl):
    Feeder(ctl).hold(DRAW_FINGERS, (0.5, 0.5), 0.5)
    ctl.execute(Command(Action.CLEAR), 10.0)
    ctl.execute(Command(Action.CLEAR), 10.0 + config.CLEAR_CONFIRM_SECONDS + 0.5)
    assert not ctl.canvas.is_empty  # Second press only re-armed it.


def test_other_command_disarms_clear(ctl):
    Feeder(ctl).hold(DRAW_FINGERS, (0.5, 0.5), 0.5)
    ctl.execute(Command(Action.CLEAR), 10.0)
    ctl.execute(Command(Action.COLOR, "Blue"), 10.5)
    ctl.execute(Command(Action.CLEAR), 11.0)
    assert not ctl.canvas.is_empty


def test_held_clear_arms_then_confirms_once(ctl):
    # Hold-to-confirm: one dwell arms Clear, a second dwell on the same item clears.
    f = Feeder(ctl)
    f.hold(DRAW_FINGERS, (0.5, 0.5), 0.5)
    f.hold(FIST_FINGERS, (0.5, 0.5), 0.3)
    clear = item_center(ctl, Action.CLEAR)
    f.hold(SELECT_FINGERS, clear, 0.25 + config.ACTION_DWELL_SECONDS + 0.1)
    assert ctl.clear_armed(f.t) and not ctl.canvas.is_empty  # Armed only.
    f.hold(SELECT_FINGERS, clear, config.ACTION_DWELL_SECONDS)
    assert ctl.canvas.is_empty and ctl.message.startswith("Canvas cleared")
    ctl.execute(Command(Action.UNDO), f.t)  # Restore, then keep holding a long time:
    f.hold(SELECT_FINGERS, clear, 10.0)
    assert not ctl.canvas.is_empty  # Latched after clearing; never clears again.
    assert ctl.executed.count(Command(Action.CLEAR)) == 2  # Arm + confirm.


def test_gesture_clear_with_two_deliberate_dwells(ctl):
    f = Feeder(ctl)
    f.hold(DRAW_FINGERS, (0.5, 0.5), 0.5)
    clear = item_center(ctl, Action.CLEAR)
    f.hold(SELECT_FINGERS, clear, 0.25 + config.ACTION_DWELL_SECONDS + 0.1)
    assert ctl.clear_armed(f.t)
    armed_at = f.t
    f.frame(SELECT_FINGERS, (0.5, 0.6))  # Leave and come back: also confirms...
    f.hold(SELECT_FINGERS, clear, config.ACTION_DWELL_SECONDS * 0.5)
    assert not ctl.canvas.is_empty  # ...but only after a full second dwell.
    f.hold(SELECT_FINGERS, clear, config.ACTION_DWELL_SECONDS * 0.6)
    assert f.t - armed_at < config.CLEAR_CONFIRM_SECONDS  # Still inside the confirm window.
    assert ctl.canvas.is_empty
    assert ctl.message.startswith("Canvas cleared")


# --- save --------------------------------------------------------------------------------

def test_save_with_empty_canvas_does_not_write(ctl):
    assert ctl.execute(Command(Action.SAVE), 1.0) == "Nothing to save yet"
    assert ctl.save_handler.calls == 0


def test_repeated_save_commands_are_rate_limited(ctl):
    Feeder(ctl).hold(DRAW_FINGERS, (0.5, 0.5), 0.5)
    ctl.execute(Command(Action.SAVE), 50.0)
    ctl.execute(Command(Action.SAVE), 50.2)  # e.g. key auto-repeat
    assert ctl.save_handler.calls == 1
    ctl.execute(Command(Action.SAVE), 50.0 + config.SAVE_CLEAR_COOLDOWN_SECONDS + 0.1)
    assert ctl.save_handler.calls == 2


def test_save_failure_is_reported(ctl):
    def broken():
        raise OSError("disk full")

    ctl.save_handler = broken
    Feeder(ctl).hold(DRAW_FINGERS, (0.5, 0.5), 0.5)
    assert ctl.execute(Command(Action.SAVE), 1.0) == "Save failed: disk full"


# --- mouse alternative --------------------------------------------------------------------

def test_mouse_draws_and_clicks_toolbar(ctl):
    ctl.mouse_press((0.2, 0.5), 0)
    for p in line((0.2, 0.5), (0.5, 0.6)):
        ctl.mouse_drag(p)
    ctl.mouse_release()
    assert len(ctl.canvas.strokes) == 1
    ctl.mouse_press(item_center(ctl, Action.COLOR, "White"), 1.0)
    assert ctl.color_name == "White"
    assert len(ctl.canvas.strokes) == 1  # Toolbar click is not a stroke.


def test_mouse_stroke_clamped_below_toolbar(ctl):
    ctl.mouse_press((0.3, 0.5), 0)
    ctl.mouse_drag((0.3, 0.01))
    ctl.mouse_release()
    assert min(p[1] for p in ctl.canvas.strokes[0].points) >= DRAW_MIN_Y
