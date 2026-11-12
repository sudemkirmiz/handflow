"""Reaching Undo / Clear / Save: toolbar kept clear of the image edges, steadier
pointer over the toolbar, larger targets for the long-dwell actions.

The jitter levels are synthetic stand-ins for the extra landmark noise of a
hand near the image edge; they do not replace a manual webcam check.
"""

import random
from pathlib import Path

import numpy as np
import pytest

import config
from core.air_canvas import (
    TOOLBAR_LEFT, TOOLBAR_RIGHT, TOOLBAR_TOP, Action, AirCanvasController, Command, build_toolbar,
)
from core.canvas import DrawingCanvas
from tests.gesture_helpers import DRAW_FINGERS, Feeder

POINT = DRAW_FINGERS


class Spy:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return Path("spy.png")


@pytest.fixture
def ctl():
    c = AirCanvasController(DrawingCanvas(), save_handler=Spy())
    c.canvas.begin_stroke((0.5, 0.6))
    c.canvas.add_point((0.6, 0.6))
    c.canvas.end_stroke()
    return c


def item(ctl, action, value=""):
    return next(i for i in ctl.toolbar if i.command == Command(action, value))


def center(it):
    x0, y0, x1, y1 = it.rect
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def glide(start, end, n):
    return [(start[0] + (end[0] - start[0]) * k / n, start[1] + (end[1] - start[1]) * k / n) for k in range(1, n + 1)]


# --- geometry ----------------------------------------------------------------------------------

def test_toolbar_keeps_clear_of_image_edges():
    items = build_toolbar()
    assert min(i.rect[0] for i in items) >= TOOLBAR_LEFT >= 0.10 - 1e-9
    assert max(i.rect[2] for i in items) <= TOOLBAR_RIGHT <= 0.90 + 1e-9
    assert min(i.rect[1] for i in items) >= TOOLBAR_TOP >= 0.05 - 1e-9
    save = next(i for i in items if i.command.action is Action.SAVE)
    assert 1.0 - save.rect[2] >= 0.10  # Was 0.02: the hand had to leave the image.


def test_action_targets_are_largest_and_items_do_not_overlap():
    items = build_toolbar()
    widths = {i.label: i.rect[2] - i.rect[0] for i in items}
    for action in ("Undo", "Clear", "Save"):
        assert widths[action] >= max(widths[c] for c in config.BRUSH_COLORS) * 1.2
        assert widths[action] >= 0.07  # ~45 px at 640 px wide.
    for a, b in zip(items, items[1:]):
        assert a.rect[2] < b.rect[0]


# --- steadier pointer over the toolbar -------------------------------------------------------------

def pointer_spread(ctl, target, sigma, seconds=3.0, seed=0):
    f = Feeder(ctl)
    rng = random.Random(seed)
    f.path(POINT, glide((target[0], 0.6), target, 10))
    seen = []
    for _ in range(round(seconds * 30)):
        f.frame(POINT, (target[0] + rng.gauss(0, sigma), target[1] + rng.gauss(0, sigma)))
        seen.append(ctl.pointer)
    return np.array(seen[20:]).std(axis=0).mean()


def test_pointer_is_steadier_over_toolbar_than_on_canvas():
    on_toolbar = pointer_spread(AirCanvasController(DrawingCanvas()), (0.5, 0.10), 0.006)
    on_canvas = pointer_spread(AirCanvasController(DrawingCanvas()), (0.5, 0.60), 0.006)
    assert on_toolbar < 0.8 * on_canvas


def test_drawing_latency_is_unchanged_on_canvas():
    ctl = AirCanvasController(DrawingCanvas())
    f = Feeder(ctl)
    lags = []
    for k in range(60):
        x = 0.2 + k / 30 * 0.5
        f.frame(POINT, (x, 0.6))
        lags.append(x - ctl.pointer[0])
    assert max(lags[20:]) < 0.012


# --- Undo / Clear / Save under heavy jitter ----------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.006, 0.010])
@pytest.mark.parametrize("action", [Action.UNDO, Action.SAVE])
def test_actions_fire_once_under_heavy_jitter(ctl, action, sigma):
    f = Feeder(ctl)
    rng = random.Random(4)
    target = center(item(ctl, action))
    f.path(POINT, glide((target[0], 0.6), target, 12))
    for _ in range(round(4.0 * 30)):
        f.frame(POINT, (target[0] + rng.gauss(0, sigma), target[1] + rng.gauss(0, sigma)))
    assert ctl.executed == [Command(action)]  # Fired, once, and nothing else.


@pytest.mark.parametrize("sigma", [0.006, 0.010])
def test_clear_confirms_under_heavy_jitter_without_touching_neighbours(ctl, sigma):
    f = Feeder(ctl)
    rng = random.Random(8)
    target = center(item(ctl, Action.CLEAR))
    f.path(POINT, glide((target[0], 0.6), target, 12))
    for _ in range(round(4.0 * 30)):
        f.frame(POINT, (target[0] + rng.gauss(0, sigma), target[1] + rng.gauss(0, sigma)))
    assert ctl.canvas.is_empty
    assert {c.action for c in ctl.executed} == {Action.CLEAR}


def test_hover_survives_drift_to_the_item_edge(ctl):
    """The finger slowly drifts across the edge of Save: the hovered item's
    enlarged hit box keeps it, instead of dropping the hover each time."""
    f = Feeder(ctl)
    save = item(ctl, Action.SAVE)
    y = center(save)[1]
    f.path(POINT, glide((center(save)[0], 0.6), center(save), 12))
    f.path(POINT, glide(center(save), (save.rect[2] + 0.012, y), 20))  # Slightly past the edge.
    f.hold(POINT, (save.rect[2] + 0.012, y), 1.5)
    assert ctl.executed == [Command(Action.SAVE)]


@pytest.mark.parametrize("action", [Action.UNDO, Action.CLEAR, Action.SAVE])
@pytest.mark.parametrize("dropout_rate", [0.06, 0.10, 0.20])
def test_actions_work_despite_tracking_dropouts(ctl, action, dropout_rate):
    """Before the fix, 1-frame hand drop-outs in 10 % of frames let Save fire in
    only ~22 % of 4 s attempts, because every drop-out restarted the dwell."""
    successes = 0
    for seed in range(10):
        c = AirCanvasController(DrawingCanvas(), save_handler=Spy())
        c.canvas.begin_stroke((0.5, 0.6))
        c.canvas.end_stroke()
        target = center(item(c, action))
        f = Feeder(c)
        rng = random.Random(seed)
        for _ in range(round(4.0 * 30)):
            if rng.random() < dropout_rate:
                f.frame(None, None)
            else:
                f.frame(POINT, (target[0] + rng.gauss(0, 0.004), target[1] + rng.gauss(0, 0.004)))
        successes += Command(action) in c.executed
        assert {x.action for x in c.executed} <= {action}  # Never a neighbour.
    assert successes == 10
