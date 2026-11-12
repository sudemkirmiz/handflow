"""Synthetic finger states for gesture tests, produced by the real finger counter."""

from __future__ import annotations

from core.finger_counter import Finger, FingerCountResult, FingerMeasurement, FingerState, count_fingers
from tests.synthetic_hand import make_hand

DRAW_FINGERS = count_fingers(make_hand({Finger.INDEX}))
SELECT_FINGERS = count_fingers(make_hand({Finger.INDEX, Finger.MIDDLE}))
FIST_FINGERS = count_fingers(make_hand(set()))
OPEN_FINGERS = count_fingers(make_hand(set(Finger)))
# Index half-bent -> UNCERTAIN, everything else folded.
UNCERTAIN_INDEX = count_fingers(make_hand(set(), half_bent={Finger.INDEX}))


def with_state(base: FingerCountResult, finger: Finger, state: FingerState) -> FingerCountResult:
    fingers = dict(base.fingers)
    fingers[finger] = FingerMeasurement(state, None, None)
    return FingerCountResult(valid=True, fingers=fingers)


class Feeder:
    """Feeds a controller timestamped snapshots with increasing sequence numbers."""

    def __init__(self, controller, fps: float = 30.0) -> None:
        self.controller = controller
        self.dt = 1.0 / fps
        self.t = 100.0
        self.seq = 0

    def frame(self, fingers, pointer, dt: float | None = None):
        self.seq += 1
        self.t += self.dt if dt is None else dt
        self.controller.process(self.seq, self.t, fingers, pointer)

    def hold(self, fingers, pointer, seconds: float):
        for _ in range(max(1, round(seconds / self.dt))):
            self.frame(fingers, pointer)

    def path(self, fingers, points):
        for p in points:
            self.frame(fingers, p)
