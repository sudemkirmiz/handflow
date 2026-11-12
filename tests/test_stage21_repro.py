"""Reproductions of the Stage 2 manual-test defects (gesture toolbar, unstable
fingertip, broken strokes). Each test models one failure *mechanism* with
synthetic but realistic input. They were written against the Stage 2 code
first, to confirm the root causes before anything was changed.
"""

import math
import random

import numpy as np
import pytest

import config
from core.air_canvas import Action, AirCanvasController, Command
from core.canvas import DrawingCanvas
from core.finger_counter import Finger, FingerState, count_fingers
from core.gestures import Gesture, classify_gesture
from tests.gesture_helpers import DRAW_FINGERS, SELECT_FINGERS, UNCERTAIN_INDEX, Feeder
from tests.synthetic_hand import make_hand

NOISE = UNCERTAIN_INDEX  # A real-classifier frame that comes out UNKNOWN.


def relaxed_thumb(hand: np.ndarray) -> np.ndarray:
    """Thumb resting loosely beside the index finger (not tucked across the palm),
    as in a natural pointing or 'peace' hand."""
    hand = hand.copy()
    cmc = np.array([0.025, 0.025, 0.0])
    d1 = np.array([0.55, 0.8, 0.25]); d1 /= np.linalg.norm(d1)
    d2 = np.array([0.15, 0.9, 0.42]); d2 /= np.linalg.norm(d2)
    hand[1] = cmc
    hand[2] = cmc + 0.035 * d1
    hand[3] = hand[2] + 0.030 * d1
    hand[4] = hand[3] + 0.025 * d2
    return hand


def ctl_with_spy():
    calls = []

    def save():
        calls.append(1)
        from pathlib import Path
        return Path("x.png")

    ctl = AirCanvasController(DrawingCanvas(), save_handler=save)
    return ctl, calls


def center(ctl, action, value=""):
    item = next(i for i in ctl.toolbar if i.command == Command(action, value))
    x0, y0, x1, y1 = item.rect
    return ((x0 + x1) / 2, (y0 + y1) / 2), item


# --- Root cause A: dwell restarts on a single noisy frame ----------------------------------

def test_dwell_survives_occasional_noisy_frames():
    """1 uncertain frame in every 8 (~12 %) must not prevent selecting a tool."""
    ctl, _ = ctl_with_spy()
    f = Feeder(ctl)
    target, _ = center(ctl, Action.UNDO)
    for i in range(int(3.0 * 30)):
        f.frame(NOISE if i % 8 == 7 else SELECT_FINGERS, target)
    assert Command(Action.UNDO) in ctl.executed


def test_dwell_survives_pointer_jitter_near_item_edge():
    """Fingertip jitter of a few pixels across an item's border must not restart dwell."""
    ctl, _ = ctl_with_spy()
    f = Feeder(ctl)
    (_, cy), item = center(ctl, Action.SAVE)
    rng = random.Random(1)
    edge_x = item.rect[0] + 0.004  # Resting just inside the left edge.
    ctl.canvas.begin_stroke((0.5, 0.5)); ctl.canvas.end_stroke()  # Something to save.
    for _ in range(int(3.0 * 30)):
        f.frame(SELECT_FINGERS, (edge_x + rng.gauss(0, 0.004), cy + rng.gauss(0, 0.004)))
    assert Command(Action.SAVE) in ctl.executed


# --- Root cause B: thumb must be folded, though no gesture depends on it --------------------

def test_relaxed_thumb_is_measured_as_not_folded():
    fingers = count_fingers(relaxed_thumb(make_hand({Finger.INDEX})))
    assert fingers.state_of(Finger.THUMB) is not FingerState.FOLDED
    assert fingers.state_of(Finger.INDEX) is FingerState.EXTENDED


def test_draw_and_select_accept_relaxed_thumb():
    draw = count_fingers(relaxed_thumb(make_hand({Finger.INDEX})))
    select = count_fingers(relaxed_thumb(make_hand({Finger.INDEX, Finger.MIDDLE})))
    assert classify_gesture(draw) is Gesture.DRAW
    assert classify_gesture(select) is Gesture.SELECT


# --- Root cause C: brief uncertainty during DRAW splits the stroke -------------------------------

def test_brief_uncertainty_does_not_split_stroke():
    ctl, _ = ctl_with_spy()
    f = Feeder(ctl)
    x = 0.2
    for i in range(90):  # 3 s of drawing; every 0.5 s a 5-frame (~0.17 s) uncertain burst.
        x += 0.004
        f.frame(NOISE if i % 15 >= 10 else DRAW_FINGERS, (x, 0.5 + 0.02 * math.sin(i / 8)))
    f.frame(None, None)
    assert len(ctl.canvas.strokes) == 1


# --- Root cause D: the first ~0.12 s of every stroke is dropped -----------------------------------

def test_stroke_starts_where_the_draw_gesture_started():
    ctl, _ = ctl_with_spy()
    f = Feeder(ctl)
    path = [(0.2 + 0.01 * k, 0.5) for k in range(30)]  # Moving from the very first frame.
    f.path(DRAW_FINGERS, path)
    f.frame(None, None)
    first = ctl.canvas.strokes[0].points[0]
    assert first[0] == pytest.approx(0.2, abs=0.012)


# --- Root cause E: fingertip jitter passes straight into strokes -----------------------------------

def test_stationary_fingertip_jitter_is_suppressed():
    ctl, _ = ctl_with_spy()
    f = Feeder(ctl)
    rng = random.Random(7)
    raw = [(0.5 + rng.gauss(0, 0.003), 0.5 + rng.gauss(0, 0.003)) for _ in range(90)]
    filtered = []
    for p in raw:
        f.frame(DRAW_FINGERS, p)
        filtered.append(ctl.pointer)
    raw_std = np.array(raw[15:]).std(axis=0).mean()
    assert np.array(filtered[15:]).std(axis=0).mean() < 0.45 * raw_std
    # A resting finger leaves a small dot, not a scribble.
    pts = np.array(ctl.canvas.active_stroke.points)
    assert (pts.max(axis=0) - pts.min(axis=0)).max() < 0.012
