import pytest

from core.finger_counter import Finger, FingerState, count_fingers
from core.gestures import Gesture, GestureStabilizer, classify_gesture
from tests.gesture_helpers import (
    DRAW_FINGERS, FIST_FINGERS, OPEN_FINGERS, SELECT_FINGERS, UNCERTAIN_INDEX, with_state,
)
from tests.synthetic_hand import make_hand, rotate


# --- 1. classification ------------------------------------------------------------

@pytest.mark.parametrize(
    "fingers,expected",
    [
        (DRAW_FINGERS, Gesture.DRAW),
        (SELECT_FINGERS, Gesture.SELECT),
        (FIST_FINGERS, Gesture.IDLE),
        (None, Gesture.NO_HAND),
        (OPEN_FINGERS, Gesture.UNKNOWN),
        (count_fingers(make_hand({Finger.INDEX, Finger.MIDDLE, Finger.RING})), Gesture.UNKNOWN),
        # Stage 2.1: the thumb no longer decides DRAW / SELECT ("L" pose draws).
        (count_fingers(make_hand({Finger.THUMB, Finger.INDEX})), Gesture.DRAW),
        (count_fingers(make_hand({Finger.THUMB, Finger.INDEX, Finger.MIDDLE})), Gesture.SELECT),
        (count_fingers(make_hand({Finger.THUMB})), Gesture.UNKNOWN),  # Thumbs-up is not a fist.
        (count_fingers(make_hand({Finger.PINKY})), Gesture.UNKNOWN),
    ],
    ids=["draw", "select", "fist", "no_hand", "open_hand", "three", "thumb_index", "thumb_index_middle",
         "thumbs_up", "pinky"],
)
def test_classification(fingers, expected):
    assert classify_gesture(fingers) is expected


def test_classification_survives_hand_rotation():
    hand = rotate(make_hand({Finger.INDEX, Finger.MIDDLE}), 30, -40, 75)
    assert classify_gesture(count_fingers(hand)) is Gesture.SELECT


# --- 2. uncertain / invalid rejection -------------------------------------------------

def test_half_bent_index_is_not_a_draw_gesture():
    assert UNCERTAIN_INDEX.state_of(Finger.INDEX) is FingerState.UNCERTAIN
    assert classify_gesture(UNCERTAIN_INDEX) is Gesture.UNKNOWN


@pytest.mark.parametrize("finger", [Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY])
def test_uncertain_defining_finger_rejects_draw_and_select(finger):
    assert classify_gesture(with_state(DRAW_FINGERS, finger, FingerState.UNCERTAIN)) is Gesture.UNKNOWN
    assert classify_gesture(with_state(SELECT_FINGERS, finger, FingerState.UNCERTAIN)) is Gesture.UNKNOWN


@pytest.mark.parametrize("state", [FingerState.UNCERTAIN, FingerState.EXTENDED])
def test_thumb_state_does_not_block_draw_or_select(state):
    assert classify_gesture(with_state(DRAW_FINGERS, Finger.THUMB, state)) is Gesture.DRAW
    assert classify_gesture(with_state(SELECT_FINGERS, Finger.THUMB, state)) is Gesture.SELECT


def test_uncertain_thumb_with_folded_fingers_is_still_pen_up():
    assert classify_gesture(with_state(FIST_FINGERS, Finger.THUMB, FingerState.UNCERTAIN)) is Gesture.IDLE


@pytest.mark.parametrize(
    "fingers,needle",
    [(None, "no hand"), (UNCERTAIN_INDEX, "uncertain: Index"), (OPEN_FINGERS, "pattern I-M-R-P 1111")],
)
def test_rejection_reasons(fingers, needle):
    from core.gestures import explain_gesture

    assert needle in explain_gesture(fingers)[1]


def test_invalid_landmarks_are_unknown_not_no_hand():
    assert classify_gesture(count_fingers(None)) is Gesture.UNKNOWN


# --- 3. stability -----------------------------------------------------------------

def test_new_gesture_needs_hold_time():
    st = GestureStabilizer(hold_seconds=0.1)
    assert st.update(Gesture.DRAW, 0.00) is Gesture.NO_HAND
    assert st.update(Gesture.DRAW, 0.05) is Gesture.NO_HAND
    assert st.update(Gesture.DRAW, 0.10) is Gesture.DRAW


def test_single_frame_flicker_is_ignored():
    st = GestureStabilizer(hold_seconds=0.1)
    for t in (0.0, 0.1):
        st.update(Gesture.DRAW, t)
    assert st.update(Gesture.SELECT, 0.13) is Gesture.DRAW
    assert st.update(Gesture.DRAW, 0.16) is Gesture.DRAW
    # The interrupted candidate must start over.
    assert st.update(Gesture.SELECT, 0.19) is Gesture.DRAW
    assert st.update(Gesture.SELECT, 0.25) is Gesture.DRAW
    assert st.update(Gesture.SELECT, 0.30) is Gesture.SELECT


def test_hand_loss_is_immediate():
    st = GestureStabilizer(hold_seconds=0.1)
    st.update(Gesture.DRAW, 0.0)
    st.update(Gesture.DRAW, 0.2)
    assert st.update(Gesture.NO_HAND, 0.21) is Gesture.NO_HAND


def test_per_gesture_hold_override():
    st = GestureStabilizer(hold_seconds=0.1, hold_overrides={Gesture.SELECT: 0.3})
    st.update(Gesture.SELECT, 0.0)
    assert st.update(Gesture.SELECT, 0.2) is Gesture.NO_HAND
    assert st.update(Gesture.SELECT, 0.3) is Gesture.SELECT


def test_brief_dropout_resumes_same_gesture():
    st = GestureStabilizer(hold_seconds=0.1, resume_seconds=0.15)
    st.update(Gesture.DRAW, 0.0)
    st.update(Gesture.DRAW, 0.2)
    assert st.update(Gesture.NO_HAND, 0.23) is Gesture.NO_HAND  # Pen lifts at once.
    assert st.update(Gesture.DRAW, 0.27) is Gesture.DRAW  # Back within 0.15 s: resumed.


def test_long_loss_requires_reconfirmation():
    st = GestureStabilizer(hold_seconds=0.1, resume_seconds=0.15)
    st.update(Gesture.DRAW, 0.0)
    st.update(Gesture.DRAW, 0.2)
    st.update(Gesture.NO_HAND, 0.23)
    assert st.update(Gesture.DRAW, 0.5) is Gesture.NO_HAND  # Too late: confirm again.
    assert st.update(Gesture.DRAW, 0.61) is Gesture.DRAW


def test_dropout_resume_only_for_the_same_gesture():
    st = GestureStabilizer(hold_seconds=0.1, resume_seconds=0.15)
    st.update(Gesture.SELECT, 0.0)
    st.update(Gesture.SELECT, 0.2)
    st.update(Gesture.NO_HAND, 0.23)
    assert st.update(Gesture.DRAW, 0.26) is Gesture.NO_HAND
