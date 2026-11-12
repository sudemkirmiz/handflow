"""Vector drawing model for Air Canvas (no UI toolkit, no MediaPipe).

Points are stored in normalized canvas coordinates: (0, 0) is the top-left and
(1, 1) the bottom-right of the camera frame. Thickness is a fraction of the
canvas height. Resizing the window therefore never distorts earlier strokes.

Undo history is a bounded stack of *completed* actions (a finished stroke or a
clear). The in-progress stroke is never part of the history.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

Point = tuple[float, float]
RGB = tuple[int, int, int]


class Tool(Enum):
    BRUSH = "Brush"
    ERASER = "Eraser"


@dataclass(frozen=True)
class Stroke:
    points: tuple[Point, ...]
    color: RGB
    thickness: float  # Fraction of canvas height.
    erase: bool = False


@dataclass(frozen=True)
class _StrokeAction:
    stroke: Stroke


@dataclass(frozen=True)
class _ClearAction:
    removed: tuple[Stroke, ...]


_Action = Union[_StrokeAction, _ClearAction]


class CanvasFullError(RuntimeError):
    pass


class DrawingCanvas:
    def __init__(
        self,
        color: RGB = (255, 143, 207),
        thickness: float = 0.016,
        eraser_factor: float = 2.5,
        max_strokes: int = 2000,
        max_points_per_stroke: int = 4000,
        undo_limit: int = 50,
        min_point_distance: float = 0.002,
    ) -> None:
        self.color = color
        self.thickness = thickness
        self.tool = Tool.BRUSH
        self.eraser_factor = eraser_factor
        self.max_strokes = max_strokes
        self.max_points_per_stroke = max_points_per_stroke
        self.min_point_distance = min_point_distance
        self._strokes: list[Stroke] = []
        self._history: deque[_Action] = deque(maxlen=undo_limit)
        self._active: Optional[list[Point]] = None
        self._active_style: Optional[tuple[RGB, float, bool]] = None
        # Bumped whenever committed strokes change; renderers use it as a cache key.
        self.version = 0

    # --- state ---------------------------------------------------------------
    @property
    def strokes(self) -> tuple[Stroke, ...]:
        return tuple(self._strokes)

    @property
    def is_drawing(self) -> bool:
        return self._active is not None

    @property
    def active_stroke(self) -> Optional[Stroke]:
        if self._active is None or self._active_style is None:
            return None
        color, thickness, erase = self._active_style
        return Stroke(tuple(self._active), color, thickness, erase)

    @property
    def is_empty(self) -> bool:
        return not self._strokes and self._active is None

    @property
    def can_undo(self) -> bool:
        return bool(self._history) or self._active is not None

    @property
    def effective_thickness(self) -> float:
        return self.thickness * (self.eraser_factor if self.tool is Tool.ERASER else 1.0)

    # --- drawing -------------------------------------------------------------
    def begin_stroke(self, point: Point) -> None:
        if self._active is not None:
            self.end_stroke()
        if len(self._strokes) >= self.max_strokes:
            raise CanvasFullError("Canvas is full — undo or clear to keep drawing")
        self._active = [_clamp(point)]
        self._active_style = (self.color, self.effective_thickness, self.tool is Tool.ERASER)

    def add_point(self, point: Point) -> None:
        if self._active is None:
            self.begin_stroke(point)
            return
        point = _clamp(point)
        last = self._active[-1]
        if math.dist(last, point) < self.min_point_distance:
            return
        if len(self._active) >= self.max_points_per_stroke:
            self.end_stroke()  # Continue seamlessly as a new stroke.
            self.begin_stroke(last)
        self._active.append(point)

    def end_stroke(self) -> Optional[Stroke]:
        """Commit the in-progress stroke (a single point becomes a dot)."""
        stroke = self.active_stroke
        self._active = None
        self._active_style = None
        if stroke is None:
            return None
        self._strokes.append(stroke)
        self._history.append(_StrokeAction(stroke))
        self.version += 1
        return stroke

    # --- editing -------------------------------------------------------------
    def undo(self) -> bool:
        if self._active is not None:
            self.end_stroke()  # Only completed strokes are undone.
        if not self._history:
            return False
        action = self._history.pop()
        if isinstance(action, _StrokeAction):
            # LIFO history guarantees the stroke is the most recent one.
            if self._strokes and self._strokes[-1] is action.stroke:
                self._strokes.pop()
        else:
            self._strokes = list(action.removed) + self._strokes
        self.version += 1
        return True

    def clear(self) -> bool:
        """Remove everything. Undoable, so a clear is never destructive."""
        if self._active is not None:
            self.end_stroke()
        if not self._strokes:
            return False
        self._history.append(_ClearAction(tuple(self._strokes)))
        self._strokes = []
        self.version += 1
        return True

    def set_color(self, color: RGB) -> None:
        self.color = color
        self.tool = Tool.BRUSH

    def set_thickness(self, thickness: float) -> None:
        self.thickness = thickness

    def set_tool(self, tool: Tool) -> None:
        self.tool = tool


def _clamp(point: Point) -> Point:
    x, y = point
    return (min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y))))
