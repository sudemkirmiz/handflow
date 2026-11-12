"""FingerTracker: finger states over a live stream with MediaPipe-like noise.

Noise model (an estimate, not a measurement of MediaPipe): sigma 2 mm in x/y and
4 mm in depth on every world landmark, every frame. Poses are synthetic; real
hands and cameras still need a manual check.
"""

import numpy as np
import pytest

from core.finger_counter import Finger, FingerState, FingerTracker, Thresholds, count_fingers
from core.processing import FrameProcessor
from tests.synthetic_hand import add_noise, make_hand, make_posed_hand, rotate

F = Finger
FOLD = (70.0, 100.0, 70.0)
OPEN_STRETCHED = make_posed_hand({}, (0, 0), 45)
OPEN_RELAXED = make_posed_hand(
    {F.INDEX: (5, 15, 10), F.MIDDLE: (5, 15, 10), F.RING: (8, 18, 12), F.PINKY: (10, 22, 15)}, (15, 15), 35)
FIST = make_hand(set())
FIST_THUMB_SIDE = make_posed_hand({f: FOLD for f in (F.INDEX, F.MIDDLE, F.RING, F.PINKY)}, (35, 45), 10)
INDEX_ONLY = make_posed_hand({F.MIDDLE: FOLD, F.RING: FOLD, F.PINKY: FOLD, F.INDEX: (5, 15, 10)}, (35, 45), 10)
PEACE = make_posed_hand({F.RING: FOLD, F.PINKY: FOLD, F.INDEX: (5, 15, 10), F.MIDDLE: (5, 15, 10)}, (35, 45), 10)


def stream(pose, n, seed=0, tracker=None):
    rng = np.random.default_rng(seed)
    tracker = tracker or FingerTracker()
    return [tracker.update(add_noise(pose, rng)) for _ in range(n)], tracker


def share(results, count):
    steady = results[15:]  # After 0.5 s at 30 FPS.
    return sum(r.extended_count == count for r in steady) / len(steady)


@pytest.mark.parametrize(
    "pose,count,minimum",
    [
        (OPEN_STRETCHED, 5, 0.95),
        (OPEN_RELAXED, 5, 0.95),
        (FIST, 0, 1.0),
        (FIST_THUMB_SIDE, 0, 1.0),
        (INDEX_ONLY, 1, 0.98),
        (PEACE, 2, 0.98),
    ],
    ids=["open_stretched", "open_relaxed", "fist", "fist_thumb_side", "index_only", "peace"],
)
def test_counts_are_stable_under_noise(pose, count, minimum):
    results, _ = stream(pose, 300, seed=1)
    assert share(results, count) >= minimum


def test_open_hand_was_unreliable_per_frame_and_is_reliable_now():
    """The mechanism behind 'five fingers are not always recognized'."""
    rng = np.random.default_rng(2)
    per_frame = [count_fingers(add_noise(OPEN_RELAXED, rng)) for _ in range(300)]
    tracked, _ = stream(OPEN_RELAXED, 300, seed=2)
    assert share(per_frame, 5) < 0.2
    assert share(tracked, 5) > 0.95


def test_relaxed_thumb_is_extended_without_noise():
    thumb = count_fingers(OPEN_RELAXED).fingers[F.THUMB]
    assert thumb.state is FingerState.EXTENDED
    assert 1.05 <= thumb.reach_ratio < 1.10  # Was just under the old 1.10 limit.


@pytest.mark.parametrize("pose", [FIST, FIST_THUMB_SIDE, INDEX_ONLY])
def test_folded_thumbs_stay_far_below_reach_threshold(pose):
    reach = count_fingers(pose).fingers[F.THUMB].reach_ratio
    assert reach < Thresholds().thumb_folded_max_reach - 0.1


def first_correct(results, count):
    return next(i for i, r in enumerate(results) if r.extended_count == count)


def test_transitions_stay_responsive():
    tracker = FingerTracker()
    stream(OPEN_RELAXED, 30, seed=3, tracker=tracker)
    to_fist, _ = stream(FIST, 30, seed=4, tracker=tracker)
    to_open, _ = stream(OPEN_RELAXED, 30, seed=5, tracker=tracker)
    to_index, _ = stream(INDEX_ONLY, 30, seed=6, tracker=tracker)
    assert first_correct(to_fist, 0) <= 3
    assert first_correct(to_open, 5) <= 8  # About a quarter of a second.
    assert first_correct(to_index, 1) <= 8
    assert share(to_fist, 0) == 1.0  # And it stays correct once reached.


def test_rotation_invariant():
    results, _ = stream(rotate(OPEN_RELAXED, 25, -35, 70), 200, seed=7)
    assert share(results, 5) > 0.95


def test_hysteresis_keeps_state_between_thresholds():
    tracker = FingerTracker(smoothing=1.0)  # No smoothing: isolate hysteresis.
    straight = make_posed_hand({F.INDEX: (0, 0, 0)})
    between = make_posed_hand({F.INDEX: (0, 30, 28)})  # 58°: beyond the 50° limit, below 90°.
    assert tracker.update(straight).state_of(F.INDEX) is FingerState.EXTENDED
    assert tracker.update(between).state_of(F.INDEX) is FingerState.EXTENDED  # Held.
    assert count_fingers(between).state_of(F.INDEX) is FingerState.UNCERTAIN  # Per frame: unsure.
    folded = make_posed_hand({F.INDEX: FOLD})
    assert tracker.update(folded).state_of(F.INDEX) is FingerState.FOLDED  # Clear change wins.


def test_invalid_input_resets():
    tracker = FingerTracker()
    stream(OPEN_RELAXED, 20, tracker=tracker)
    assert tracker.update(None).valid is False
    first = tracker.update(FIST)  # No memory of the open hand.
    assert first.extended_count == 0


def test_sudden_jump_reinitializes_instead_of_smoothing():
    tracker = FingerTracker()
    stream(OPEN_RELAXED, 20, tracker=tracker)
    elsewhere = OPEN_RELAXED + np.array([0.2, 0.0, 0.0])  # E.g. a different hand.
    assert tracker.update(elsewhere).extended_count == 5


def test_processor_resets_finger_state_when_hand_is_lost():
    from core.hand_tracker import HandTrackingResult, Landmark, TrackedHand
    from tests.test_processing import FakeCamera

    def hand(points):
        lms = tuple(Landmark(*p) for p in points)
        return TrackedHand(lms, lms, "Right", 0.9)

    class Sequence:
        def __init__(self, hands):
            self.hands = list(hands)

        def detect(self, frame):
            h = self.hands.pop(0)
            return HandTrackingResult(0, (h,) if h else ())

        def close(self):
            pass

    hands = [hand(OPEN_RELAXED)] * 10 + [None] + [hand(FIST)]
    proc = FrameProcessor(FakeCamera(can_open=True), lambda: Sequence(hands))
    proc.load_tracker()
    frame = np.zeros((48, 64, 3), np.uint8)
    for _ in range(10):
        snap = proc.process_frame(frame)
    assert snap.fingers.extended_count == 5
    assert proc.process_frame(frame).fingers is None
    assert proc.process_frame(frame).fingers.extended_count == 0  # Fresh start, no smoothing lag.
