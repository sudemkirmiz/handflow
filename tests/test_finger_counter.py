import math

import numpy as np
import pytest

from core.finger_counter import Finger, FingerState, count_fingers, joint_angle
from tests.synthetic_hand import make_hand, rotate

ALL = set(Finger)


def states(result):
    return {f: result.state_of(f) for f in Finger}


def test_fully_open_hand_counts_five():
    result = count_fingers(make_hand(ALL))
    assert result.valid
    assert result.extended_count == 5
    assert all(s is FingerState.EXTENDED for s in states(result).values())


def test_closed_fist_counts_zero():
    result = count_fingers(make_hand(set()))
    assert result.valid
    assert result.extended_count == 0
    assert all(s is FingerState.FOLDED for s in states(result).values())


def test_index_finger_only():
    result = count_fingers(make_hand({Finger.INDEX}))
    assert result.extended_count == 1
    assert result.state_of(Finger.INDEX) is FingerState.EXTENDED
    for f in ALL - {Finger.INDEX}:
        assert result.state_of(f) is FingerState.FOLDED


@pytest.mark.parametrize(
    "pair",
    [{Finger.INDEX, Finger.MIDDLE}, {Finger.THUMB, Finger.INDEX}, {Finger.THUMB, Finger.PINKY}],
)
def test_two_extended_fingers(pair):
    result = count_fingers(make_hand(pair))
    assert result.extended_count == 2
    assert {f for f, s in states(result).items() if s is FingerState.EXTENDED} == pair


@pytest.mark.parametrize("angles", [(0, 0, 90), (0, 0, 180), (35, -50, 120), (-60, 25, -30), (80, 10, 0)])
def test_counting_is_rotation_invariant(angles):
    hand = rotate(make_hand({Finger.THUMB, Finger.INDEX, Finger.MIDDLE}), *angles)
    assert count_fingers(hand).extended_count == 3


def test_mirrored_left_hand_is_counted_the_same():
    left = make_hand({Finger.INDEX, Finger.PINKY}) * np.array([-1.0, 1.0, 1.0])
    assert count_fingers(left).extended_count == 2


def test_half_bent_finger_is_uncertain_not_guessed():
    result = count_fingers(make_hand({Finger.MIDDLE}, half_bent={Finger.INDEX}))
    assert result.state_of(Finger.INDEX) is FingerState.UNCERTAIN
    assert result.uncertain_count == 1
    assert result.extended_count == 1  # Only confident extensions are counted.


def test_accepts_landmark_objects():
    from core.hand_tracker import Landmark

    pts = [Landmark(*p) for p in make_hand(ALL)]
    assert count_fingers(pts).extended_count == 5


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        make_hand(ALL)[:20],
        np.vstack([make_hand(ALL), [[0, 0, 0]]]),
        [["a", "b", "c"]] * 21,
        [(0.0, 0.0)] * 21,
    ],
    ids=["none", "empty", "too_few", "too_many", "non_numeric", "2d_points"],
)
def test_invalid_landmarks_are_reported_not_counted(bad):
    result = count_fingers(bad)
    assert not result.valid
    assert result.extended_count is None
    assert result.uncertain_count == 5
    assert result.reason


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_non_finite_landmarks_are_invalid(value):
    pts = make_hand(ALL)
    pts[8, 1] = value
    assert count_fingers(pts).valid is False


def test_degenerate_finger_is_uncertain():
    pts = make_hand(ALL)
    pts[6] = pts[5]  # Zero-length index segment.
    result = count_fingers(pts)
    assert result.valid
    assert result.state_of(Finger.INDEX) is FingerState.UNCERTAIN


def test_joint_angle_basics():
    a, b = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    assert joint_angle(a, b, np.array([2.0, 0.0, 0.0])) == pytest.approx(180.0)
    assert joint_angle(a, b, np.array([1.0, 1.0, 0.0])) == pytest.approx(90.0)
    assert joint_angle(a, a, b) is None
