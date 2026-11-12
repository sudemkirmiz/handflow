from types import SimpleNamespace

import numpy as np
import pytest

from core.hand_tracker import (
    HAND_CONNECTIONS,
    HandTracker,
    MonotonicTimestamper,
    resolve_handedness,
    verify_model_file,
)


class FakeClock:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


def fake_points(n=21):
    return [SimpleNamespace(x=i / 30, y=i / 40, z=-i / 100) for i in range(n)]


def fake_raw(n_points=21, label="Right", score=0.93):
    return SimpleNamespace(
        hand_landmarks=[fake_points(n_points)],
        hand_world_landmarks=[fake_points(n_points)],
        handedness=[[SimpleNamespace(category_name=label, score=score)]],
    )


class FakeLandmarker:
    def __init__(self, results=None):
        self.timestamps = []
        self.results = results or []
        self.closed = 0

    def detect_for_video(self, image, timestamp_ms):
        if self.timestamps and timestamp_ms <= self.timestamps[-1]:
            raise ValueError("Input timestamp must be monotonically increasing.")
        self.timestamps.append(timestamp_ms)
        return self.results.pop(0) if self.results else SimpleNamespace(
            hand_landmarks=[], hand_world_landmarks=[], handedness=[]
        )

    def close(self):
        self.closed += 1


# --- timestamps ---------------------------------------------------------------

def test_timestamps_strictly_increase_even_if_clock_stalls_or_goes_back():
    # start=10.0, then: same ms twice, a backwards jump, then a real advance.
    clock = FakeClock([10.0, 10.0, 10.0, 9.5, 10.0004, 10.5])
    ts = MonotonicTimestamper(clock)
    values = [ts.next() for _ in range(5)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)
    assert values[-1] == 500


def test_tracker_feeds_monotonic_timestamps_to_mediapipe():
    landmarker = FakeLandmarker()
    tracker = HandTracker(landmarker, timestamper=MonotonicTimestamper(FakeClock([0.0])))
    frame = np.zeros((48, 64, 3), np.uint8)
    for _ in range(25):  # Frozen clock: every frame lands in the same millisecond.
        tracker.detect(frame)
    assert landmarker.timestamps == list(range(25))


# --- result conversion ----------------------------------------------------------

def test_convert_result_keeps_model_output():
    tracker = HandTracker(FakeLandmarker(), input_is_mirrored=True)
    result = tracker.convert_result(fake_raw(label="Left", score=0.81), 42)
    hand = result.primary_hand
    assert result.timestamp_ms == 42
    assert len(hand.landmarks) == 21 and len(hand.world_landmarks) == 21
    assert hand.handedness == "Left"
    assert hand.handedness_score == pytest.approx(0.81)


def test_unmirrored_input_swaps_handedness():
    tracker = HandTracker(FakeLandmarker(), input_is_mirrored=False)
    assert tracker.convert_result(fake_raw(label="Left"), 1).primary_hand.handedness == "Right"


@pytest.mark.parametrize(
    "label,mirrored,expected",
    [("Left", True, "Left"), ("Right", True, "Right"), ("Left", False, "Right"),
     ("Right", False, "Left"), (None, True, None), ("Unknown", False, None)],
)
def test_resolve_handedness(label, mirrored, expected):
    assert resolve_handedness(label, mirrored) == expected


def test_partial_skeleton_is_dropped():
    tracker = HandTracker(FakeLandmarker())
    assert tracker.convert_result(fake_raw(n_points=15), 1).primary_hand is None


def test_empty_result_means_no_hand():
    tracker = HandTracker(FakeLandmarker())
    empty = SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[], handedness=[])
    assert tracker.convert_result(empty, 1).hands == ()


def test_missing_handedness_is_none_not_invented():
    tracker = HandTracker(FakeLandmarker())
    raw = SimpleNamespace(hand_landmarks=[fake_points()], hand_world_landmarks=[], handedness=[])
    hand = tracker.convert_result(raw, 1).primary_hand
    assert hand.handedness is None and hand.handedness_score is None
    assert hand.world_landmarks == ()


def test_close_is_idempotent_and_blocks_further_use():
    landmarker = FakeLandmarker()
    tracker = HandTracker(landmarker)
    tracker.close()
    tracker.close()
    assert landmarker.closed == 1
    with pytest.raises(RuntimeError):
        tracker.detect(np.zeros((4, 4, 3), np.uint8))


def test_missing_model_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        HandTracker.from_model(tmp_path / "missing.task")


def test_corrupted_model_is_rejected(tmp_path):
    bad = tmp_path / "hand_landmarker.task"
    bad.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="checksum"):
        verify_model_file(bad, "0" * 64)


def test_bundled_model_matches_documented_checksum():
    import config

    verify_model_file(config.MODEL_PATH, config.MODEL_SHA256)  # Raises on mismatch.


# --- topology -------------------------------------------------------------------

def test_connections_match_official_mediapipe_topology():
    vision = pytest.importorskip("mediapipe.tasks.python.vision")
    official = {(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS}
    assert set(HAND_CONNECTIONS) == official
    assert len(HAND_CONNECTIONS) == 21
    assert {i for c in HAND_CONNECTIONS for i in c} == set(range(21))
