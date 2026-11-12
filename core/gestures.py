"""Rule-based gesture recognition on top of Stage 1 finger states.

No trained classifier: a gesture is a pattern of ``FingerState`` values. Any
UNCERTAIN finger makes the gesture UNKNOWN, so a drawing gesture is never
inferred from unreliable data.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from core.finger_counter import Finger, FingerCountResult, FingerState


class Gesture(Enum):
    DRAW = "Draw"  # Index finger only.
    SELECT = "Select"  # Index + middle.
    IDLE = "Pen up"  # Closed fist.
    NO_HAND = "No hand"
    UNKNOWN = "Unknown"  # Uncertain fingers or any other pattern.


_E, _F, _U = FingerState.EXTENDED, FingerState.FOLDED, FingerState.UNCERTAIN
_LONG_FINGERS = (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY)

# DRAW and SELECT are defined by the four long fingers only. The thumb is the
# least reliable finger to measure, and a relaxed thumb beside the index is
# natural when pointing; no gesture needs the thumb to tell DRAW from SELECT.
# (Stage 2 required a folded thumb, which rejected those natural poses.)
_PATTERNS: dict[tuple[FingerState, ...], Gesture] = {
    # (index, middle, ring, pinky)
    (_E, _F, _F, _F): Gesture.DRAW,
    (_E, _E, _F, _F): Gesture.SELECT,
    (_F, _F, _F, _F): Gesture.IDLE,
}
_SHORT = {_E: "1", _F: "0", _U: "?"}


def explain_gesture(fingers: Optional[FingerCountResult]) -> tuple[Gesture, str]:
    """Gesture plus a short reason, used for diagnostics when it is UNKNOWN."""
    if fingers is None:
        return Gesture.NO_HAND, "no hand"
    if not fingers.valid:
        return Gesture.UNKNOWN, fingers.reason or "invalid landmarks"
    long_states = tuple(fingers.state_of(f) for f in _LONG_FINGERS)
    unsure = [f.value for f in _LONG_FINGERS if fingers.state_of(f) is _U]
    if unsure:
        return Gesture.UNKNOWN, "uncertain: " + ", ".join(unsure)
    gesture = _PATTERNS.get(long_states)
    if gesture is None:
        return Gesture.UNKNOWN, "pattern I-M-R-P " + "".join(_SHORT[s] for s in long_states)
    if gesture is Gesture.IDLE and fingers.state_of(Finger.THUMB) is _E:
        return Gesture.UNKNOWN, "thumb up (not a fist)"  # Still pen-up, just not IDLE.
    return gesture, ""


def classify_gesture(fingers: Optional[FingerCountResult]) -> Gesture:
    return explain_gesture(fingers)[0]


class GestureStabilizer:
    """Debounces gestures: a new one must persist for a hold time to take effect.

    Losing the hand (NO_HAND) takes effect immediately so the pen lifts at once.
    If the hand comes back within ``resume_seconds`` showing the gesture it had
    before (a tracking drop-out, common near the image edges), that gesture is
    restored at once instead of having to be confirmed again.
    """

    def __init__(
        self,
        hold_seconds: float = 0.12,
        hold_overrides: Optional[dict[Gesture, float]] = None,
        resume_seconds: float = 0.15,
    ) -> None:
        self._hold = hold_seconds
        self._overrides = hold_overrides or {}
        self._resume = resume_seconds
        self.stable = Gesture.NO_HAND
        self._candidate: Optional[Gesture] = None
        self._candidate_since = 0.0
        self._before_loss: Optional[Gesture] = None
        self._lost_at = 0.0

    def update(self, raw: Gesture, timestamp: float) -> Gesture:
        if raw is Gesture.NO_HAND:
            if self.stable is not Gesture.NO_HAND:
                self._before_loss, self._lost_at = self.stable, timestamp
            self.stable = Gesture.NO_HAND
            self._candidate = None
            return self.stable
        if (
            self.stable is Gesture.NO_HAND
            and raw is self._before_loss
            and timestamp - self._lost_at <= self._resume
        ):
            self.stable = raw  # Brief drop-out: same gesture continues.
            self._before_loss = None
            self._candidate = None
            return self.stable
        self._before_loss = None
        if raw is self.stable:
            self._candidate = None
            return self.stable
        if raw is not self._candidate:
            self._candidate = raw
            self._candidate_since = timestamp
        if timestamp - self._candidate_since >= self._overrides.get(raw, self._hold):
            self.stable = raw
            self._candidate = None
        return self.stable

    def reset(self) -> None:
        self.stable = Gesture.NO_HAND
        self._candidate = None
        self._before_loss = None
