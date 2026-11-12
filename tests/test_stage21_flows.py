"""Stage 2.1 end-to-end controller flows under realistic noise (no GUI).

Noise model: every 8th frame comes out UNKNOWN from the real finger counter
(~12 %), and the fingertip jitters with sigma 0.003 (~2 px at 640x480). These
are synthetic stand-ins for webcam conditions; they do not prove physical
behavior, but they reproduce the mechanisms that broke it.
"""

import math
import random
from pathlib import Path

import pytest

import config
from core.air_canvas import DRAW_MIN_Y, Action, AirCanvasController, Command
from core.canvas import DrawingCanvas, Tool
from core.gestures import Gesture
from tests.gesture_helpers import DRAW_FINGERS, FIST_FINGERS, SELECT_FINGERS, UNCERTAIN_INDEX, Feeder

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


def noisy_hold(f, fingers, point, seconds, seed=0, noise_every=8, sigma=0.003):
    rng = random.Random(seed)
    for i in range(round(seconds * 30)):
        jittered = (point[0] + rng.gauss(0, sigma), point[1] + rng.gauss(0, sigma))
        f.frame(NOISE if noise_every and i % noise_every == noise_every - 1 else fingers, jittered)


def noisy_path(f, fingers, points, seed=0, noise_every=8, sigma=0.003):
    rng = random.Random(seed)
    for i, (x, y) in enumerate(points):
        f.frame(NOISE if noise_every and i % noise_every == noise_every - 1 else fingers,
                (x + rng.gauss(0, sigma), y + rng.gauss(0, sigma)))


def center(item):
    x0, y0, x1, y1 = item.rect
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def item_for(ctl, action, value=""):
    return next(i for i in ctl.toolbar if i.command == Command(action, value))


def expected_log(item):
    """Size cycling is logged as the size it resolved to (Medium -> Thick)."""
    if item.command.action is Action.CYCLE_SIZE:
        return Command(Action.SIZE, "Thick")
    return item.command


def make_drawing(ctl):
    ctl.canvas.begin_stroke((0.4, 0.5))
    ctl.canvas.add_point((0.6, 0.5))
    ctl.canvas.end_stroke()


# --- 1/2. SELECT reaches every toolbar item; every command has an activation path --------------

@pytest.mark.parametrize("index", range(11))
def test_every_toolbar_item_activates_by_gesture_under_noise(ctl, index):
    item = ctl.toolbar[index]
    make_drawing(ctl)
    f = Feeder(ctl)
    noisy_hold(f, SELECT_FINGERS, center(item), 2.0, seed=index)
    assert expected_log(item) in ctl.executed
    if item.command.action is Action.CLEAR:
        assert ctl.clear_armed(f.t)  # First activation only arms Clear.
    if item.command.action is Action.SAVE:
        assert ctl.save_handler.calls == 1


def test_clear_by_gesture_needs_two_deliberate_dwells_under_noise(ctl):
    make_drawing(ctl)
    f = Feeder(ctl)
    clear = item_for(ctl, Action.CLEAR)
    # Noise pauses the timer on ~1/8 of frames, so allow for that in the hold times.
    noisy_hold(f, SELECT_FINGERS, center(clear), 0.3 + config.ACTION_DWELL_SECONDS * 1.3, seed=1)
    assert ctl.clear_armed(f.t) and not ctl.canvas.is_empty
    noisy_hold(f, SELECT_FINGERS, (0.5, 0.6), 0.3, seed=2)  # Move off to re-arm.
    cooldown = config.COMMAND_COOLDOWN_SECONDS + config.SAVE_CLEAR_COOLDOWN_SECONDS
    noisy_hold(f, SELECT_FINGERS, center(clear), cooldown + config.ACTION_DWELL_SECONDS * 1.4, seed=3)
    assert ctl.canvas.is_empty


def test_gesture_color_and_eraser_apply_to_next_strokes(ctl):
    f = Feeder(ctl)
    noisy_hold(f, SELECT_FINGERS, center(item_for(ctl, Action.COLOR, "Mint")), 1.6)
    noisy_path(f, DRAW_FINGERS, [(0.3 + 0.01 * k, 0.6) for k in range(25)], seed=4)
    noisy_hold(f, FIST_FINGERS, (0.55, 0.6), 0.6)
    noisy_hold(f, SELECT_FINGERS, center(item_for(ctl, Action.ERASER)), 2.2, seed=5)
    noisy_path(f, DRAW_FINGERS, [(0.3 + 0.01 * k, 0.7) for k in range(25)], seed=6)
    noisy_hold(f, FIST_FINGERS, (0.55, 0.7), 0.6)
    first, second = ctl.canvas.strokes
    assert first.color == config.hex_to_rgb(config.BRUSH_COLORS["Mint"]) and not first.erase
    assert second.erase and ctl.canvas.tool is Tool.ERASER


# --- 3. mouse toolbar controls --------------------------------------------------------------------

@pytest.mark.parametrize("index", range(11))
def test_every_toolbar_item_works_by_mouse_click(ctl, index):
    item = ctl.toolbar[index]
    make_drawing(ctl)
    ctl.mouse_press(center(item), now=100.0)
    ctl.mouse_release()
    assert ctl.executed == [expected_log(item)]
    assert len(ctl.canvas.strokes) <= 1  # A click on the toolbar never starts a stroke.


# --- 4/11. dwell, cooldown, no repeats from a held gesture -------------------------------------------

@pytest.mark.parametrize("action", [Action.UNDO, Action.SAVE, Action.CLEAR])
def test_held_noisy_gesture_fires_once(ctl, action):
    for _ in range(3):
        make_drawing(ctl)
    f = Feeder(ctl)
    noisy_hold(f, SELECT_FINGERS, center(item_for(ctl, action)), 8.0, seed=9)
    if action is Action.CLEAR:
        assert ctl.executed.count(Command(action)) == 2  # Arm, then hold-to-confirm; once.
        assert ctl.canvas.is_empty
    else:
        assert ctl.executed.count(Command(action)) == 1
    if action is Action.SAVE:
        assert ctl.save_handler.calls == 1


def test_dwell_pause_does_not_count_paused_time(ctl):
    f = Feeder(ctl)
    target = center(item_for(ctl, Action.UNDO))
    f.hold(SELECT_FINGERS, target, 0.25)  # Stable SELECT.
    f.hold(SELECT_FINGERS, target, config.ACTION_DWELL_SECONDS * 0.5)
    progress = ctl.dwell_progress
    for _ in range(6):  # 0.2 s of uncertain frames: paused, not reset, not advanced.
        f.frame(NOISE, target)
    assert ctl.dwell_progress == pytest.approx(progress)
    assert ctl.hovered is not None
    f.hold(SELECT_FINGERS, target, config.ACTION_DWELL_SECONDS * 0.6)
    assert Command(Action.UNDO) in ctl.executed


def test_long_miss_resets_dwell(ctl):
    f = Feeder(ctl)
    target = center(item_for(ctl, Action.UNDO))
    f.hold(SELECT_FINGERS, target, 0.25 + config.DWELL_SECONDS * 0.6)
    f.hold(SELECT_FINGERS, (0.5, 0.6), config.DWELL_GRACE_SECONDS + 0.1)  # Clearly elsewhere.
    assert ctl.hovered is None and ctl.dwell_progress == 0.0


def test_border_jitter_does_not_flip_between_neighbours(ctl):
    f = Feeder(ctl)
    blue, mint = item_for(ctl, Action.COLOR, "Blue"), item_for(ctl, Action.COLOR, "Mint")
    border = ((blue.rect[2] + mint.rect[0]) / 2, (blue.rect[1] + blue.rect[3]) / 2)
    f.hold(SELECT_FINGERS, center(blue), 0.3)
    noisy_hold(f, SELECT_FINGERS, (blue.rect[2] - 0.002, border[1]), 1.5, noise_every=0, sigma=0.004)
    assert ctl.executed == [Command(Action.COLOR, "Blue")]


# --- 5/6/7. drawing continuity -------------------------------------------------------------------

def wave(n, y=0.5, x0=0.2, dx=0.006):
    return [(x0 + dx * k, y + 0.03 * math.sin(k / 6)) for k in range(n)]


def test_noisy_drawing_is_one_continuous_stroke(ctl):
    f = Feeder(ctl)
    noisy_path(f, DRAW_FINGERS, wave(100))
    f.frame(None, None)
    (stroke,) = ctl.canvas.strokes
    steps = [math.dist(a, b) for a, b in zip(stroke.points, stroke.points[1:])]
    assert max(steps) < 0.03  # Segments between consecutive accepted positions only.
    assert stroke.points[0][0] == pytest.approx(0.2, abs=0.015)


def test_brief_select_flicker_during_draw_does_not_split(ctl):
    f = Feeder(ctl)
    pts = wave(60)
    for i, p in enumerate(pts):
        f.frame(SELECT_FINGERS if i in (20, 21, 40) else DRAW_FINGERS, p)  # < SELECT hold time.
    f.frame(None, None)
    assert len(ctl.canvas.strokes) == 1


def test_prolonged_uncertainty_ends_stroke_and_no_line_on_resume(ctl):
    f = Feeder(ctl)
    f.path(DRAW_FINGERS, wave(30))
    for k in range(round((config.UNCERTAIN_GRACE_SECONDS + 0.2) * 30)):
        f.frame(NOISE, (0.4 + 0.01 * k, 0.75))  # Hand moves while uncertain.
    assert ctl.gesture is Gesture.UNKNOWN and not ctl.canvas.is_drawing
    f.path(DRAW_FINGERS, wave(20, y=0.8, x0=0.6))
    f.frame(None, None)
    first, second = ctl.canvas.strokes
    assert math.dist(first.points[-1], second.points[0]) > 0.1  # Separate strokes, no bridge.


def test_moving_during_brief_pause_starts_new_stroke(ctl):
    f = Feeder(ctl)
    f.path(DRAW_FINGERS, wave(20))
    for k in range(5):  # 0.17 s uncertain, but the finger travels 0.08.
        f.frame(NOISE, (0.32 + 0.02 * k, 0.5))
    f.path(DRAW_FINGERS, [(0.42 + 0.005 * k, 0.5) for k in range(15)])
    f.frame(None, None)
    assert len(ctl.canvas.strokes) == 2


def test_select_ends_stroke(ctl):
    f = Feeder(ctl)
    f.path(DRAW_FINGERS, wave(20))
    f.hold(SELECT_FINGERS, (0.4, 0.5), config.SELECT_HOLD_SECONDS + 0.1)
    assert ctl.gesture is Gesture.SELECT and not ctl.canvas.is_drawing


# --- 8. hand loss and reacquisition ---------------------------------------------------------------

def test_reacquired_hand_starts_exactly_at_new_position(ctl):
    f = Feeder(ctl)
    f.path(DRAW_FINGERS, wave(20))
    f.frame(None, None)
    f.frame(None, None)
    f.path(DRAW_FINGERS, [(0.7 + 0.005 * k, 0.8) for k in range(15)])
    f.frame(None, None)
    second = ctl.canvas.strokes[1]
    # The filter was reset: no smoothing pull from the old position.
    assert second.points[0] == pytest.approx((0.7, 0.8), abs=1e-9)


def test_drawing_never_enters_toolbar_band_under_noise(ctl):
    f = Feeder(ctl)
    noisy_path(f, DRAW_FINGERS, [(0.2 + 0.01 * k, 0.12) for k in range(50)])
    f.frame(None, None)
    assert all(p[1] >= DRAW_MIN_Y for s in ctl.canvas.strokes for p in s.points)


# --- 10. latency of the pointer pipeline ---------------------------------------------------------

def test_pointer_lag_is_small_while_moving(ctl):
    f = Feeder(ctl)
    lags = []
    for k in range(60):
        x = 0.2 + k / 30 * 0.5  # 0.5 frame-widths per second.
        f.frame(DRAW_FINGERS, (x, 0.5))
        lags.append(x - ctl.pointer[0])
    assert max(lags[20:]) < 0.012
