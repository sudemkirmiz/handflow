"""One Euro filter for the fingertip pointer (Casiez, Roussel & Vogel, CHI 2012).

An exponential low-pass filter whose cutoff rises with speed: strong smoothing
when the finger is (nearly) still, so jitter disappears, and little lag when it
moves fast. Timestamps come from the tracking snapshots, so the result doesn't
depend on how often the UI happens to poll.
"""

from __future__ import annotations

import math
from typing import Optional

Point = tuple[float, float]


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: Optional[float] = None
        self._dx = 0.0
        self._t: Optional[float] = None

    def reset(self) -> None:
        self._x, self._dx, self._t = None, 0.0, None

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None or t <= self._t:
            self._x, self._dx, self._t = x, 0.0, t
            return x
        dt = t - self._t
        self._t = t
        dx = (x - self._x) / dt
        self._dx += _alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x += _alpha(cutoff, dt) * (x - self._x)
        return self._x


class PointFilter:
    """Filters x and y independently; ``reset()`` on tracking loss or jumps."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()

    def set_params(self, min_cutoff: float, beta: float) -> None:
        """Change the smoothing strength on the fly (state is kept)."""
        for f in (self._fx, self._fy):
            f.min_cutoff, f.beta = min_cutoff, beta

    def __call__(self, point: Point, t: float) -> Point:
        return (self._fx(point[0], t), self._fy(point[1], t))
