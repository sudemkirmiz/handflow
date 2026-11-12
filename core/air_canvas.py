"""Air Canvas controller: turns stable gestures into drawing and toolbar commands.

Input per tracking snapshot: sequence number, timestamp, finger states and the
index fingertip (normalized coordinates of the displayed, mirrored frame).

Rules
-----
* The fingertip is smoothed with a One Euro filter (reset on gaps and jumps).
* DRAW (stable *and* in the current frame) adds points to a stroke. The frames
  that confirmed DRAW are drawn too, so strokes start where the gesture started.
* Brief uncertainty (UNKNOWN while the hand is tracked) only *pauses* the pen
  for up to ``uncertain_grace_seconds``; the stroke continues only if the
  fingertip barely moved meanwhile. Longer uncertainty, another stable
  gesture or a lost hand lifts the pen.
* A long tracking gap or a big jump starts a new stroke instead of drawing a
  long accidental line.
* One finger does everything. The pointer's *zone* is checked first: in the
  toolbar zone a pointing hand (index only, or index + middle) never draws, it
  hovers; resting on an item for its dwell time runs it once (0.5 s for colors,
  size and eraser, 1.0 s for Undo, Clear and Save). The dwell only advances
  while the finger is nearly still, so sweeping across the toolbar selects
  nothing. Brief misses pause the dwell timer instead of restarting it.
* Below the toolbar is a buffer: no hovering, and no new strokes start there,
  so leaving the toolbar never leaves a mark and returning to the canvas
  starts a clean stroke.
  The item is then *latched* until the pointer leaves it, so a held gesture
  can never repeat Save. A cooldown separates consecutive commands.
* Clear needs two activations within ``clear_confirm_seconds`` and stays
  undoable afterwards.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import config
from core.canvas import CanvasFullError, DrawingCanvas, Point, Tool
from core.finger_counter import FingerCountResult
from core.gestures import Gesture, GestureStabilizer, explain_gesture
from core.smoothing import PointFilter


class Action(Enum):
    COLOR = "color"
    SIZE = "size"
    CYCLE_SIZE = "cycle_size"
    ERASER = "eraser"
    UNDO = "undo"
    CLEAR = "clear"
    SAVE = "save"


@dataclass(frozen=True)
class Command:
    action: Action
    value: str = ""


@dataclass(frozen=True)
class ToolbarItem:
    command: Command
    label: str
    rect: tuple[float, float, float, float]  # Normalized x0, y0, x1, y1.

    def contains(self, point: Point, margin: float = 0.0) -> bool:
        x0, y0, x1, y1 = self.rect
        return x0 - margin <= point[0] <= x1 + margin and y0 - margin <= point[1] <= y1 + margin


HINT = "☝ Point to draw · Hover over a tool to select"

# Interaction regions, in normalized frame coordinates (y grows downwards):
#   y <= TOOLBAR_ZONE_BOTTOM         toolbar zone: hover/select only, never draw
#   TOOLBAR_ZONE_BOTTOM .. DRAW_START_Y   buffer: no hover, no new stroke
#   y >= DRAW_START_Y                canvas: drawing
# A stroke that is already running may continue up to DRAW_MIN_Y, so jitter
# around the canvas edge does not chop a line into tiny pieces (hysteresis).
# The toolbar keeps clear of the image edges: to point at an item near an edge,
# much of the hand (palm, thumb or pinky side) would be outside the camera
# image, and MediaPipe tracks a cropped hand worse (more jitter, brief
# drop-outs). With the old 2 % margin, pointing at Save put the hand past the
# right edge.
TOOLBAR_LEFT = 0.10
TOOLBAR_RIGHT = 0.90
TOOLBAR_TOP = 0.05
TOOLBAR_BOTTOM = 0.16  # Bottom of the toolbar items.
TOOLBAR_ZONE_BOTTOM = TOOLBAR_BOTTOM + 0.012  # Bottom of the drawn toolbar backdrop.
DRAW_MIN_Y = TOOLBAR_BOTTOM + 0.02  # Strokes never go above this line.
DRAW_START_Y = TOOLBAR_BOTTOM + 0.04  # New strokes start at or below this line.


class Zone(Enum):
    TOOLBAR = "toolbar"
    BUFFER = "buffer"
    CANVAS = "canvas"


def zone_of(point: Point) -> Zone:
    if point[1] <= TOOLBAR_ZONE_BOTTOM:
        return Zone.TOOLBAR
    if point[1] < DRAW_START_Y:
        return Zone.BUFFER
    return Zone.CANVAS


EXECUTED_LOG_LIMIT = 200
POINTING = (Gesture.DRAW, Gesture.SELECT)  # Gestures whose fingertip may hover the toolbar.
ACTIONS_NEEDING_LONG_DWELL = (Action.UNDO, Action.CLEAR, Action.SAVE)


def build_toolbar() -> tuple[ToolbarItem, ...]:
    commands = [Command(Action.COLOR, name) for name in config.BRUSH_COLORS]
    commands += [
        Command(Action.CYCLE_SIZE),
        Command(Action.ERASER),
        Command(Action.UNDO),
        Command(Action.CLEAR),
        Command(Action.SAVE),
    ]
    labels = list(config.BRUSH_COLORS) + ["Size", "Eraser", "Undo", "Clear", "Save"]
    # Relative widths: swatches are a little narrower so that Undo/Clear/Save,
    # which need a deliberate 1 s rest, keep larger targets.
    weights = [0.9 if c.action is Action.COLOR else 1.1 if c.action in ACTIONS_NEEDING_LONG_DWELL else 1.0
               for c in commands]
    gap = 0.007
    unit = (TOOLBAR_RIGHT - TOOLBAR_LEFT - gap * (len(commands) - 1)) / sum(weights)
    items = []
    x0 = TOOLBAR_LEFT
    for command, label, weight in zip(commands, labels, weights):
        x1 = x0 + weight * unit
        items.append(ToolbarItem(command, label, (x0, TOOLBAR_TOP, x1, TOOLBAR_BOTTOM)))
        x0 = x1 + gap
    return tuple(items)


def hex_rgb(hex_color: str) -> tuple[int, int, int]:
    return config.hex_to_rgb(hex_color)


SaveHandler = Callable[[], Path]


class AirCanvasController:
    def __init__(
        self,
        canvas: DrawingCanvas,
        save_handler: Optional[SaveHandler] = None,
        toolbar: Optional[tuple[ToolbarItem, ...]] = None,
        hold_seconds: float = config.GESTURE_HOLD_SECONDS,
        select_hold_seconds: float = config.SELECT_HOLD_SECONDS,
        uncertain_grace_seconds: float = config.UNCERTAIN_GRACE_SECONDS,
        dwell_seconds: float = config.DWELL_SECONDS,
        action_dwell_seconds: float = config.ACTION_DWELL_SECONDS,
        hover_max_speed: float = config.HOVER_MAX_SPEED,
        dwell_grace_seconds: float = config.DWELL_GRACE_SECONDS,
        hover_margin: float = config.HOVER_MARGIN,
        cooldown_seconds: float = config.COMMAND_COOLDOWN_SECONDS,
        save_clear_cooldown_seconds: float = config.SAVE_CLEAR_COOLDOWN_SECONDS,
        clear_confirm_seconds: float = config.CLEAR_CONFIRM_SECONDS,
        max_gap_seconds: float = config.STROKE_MAX_GAP_SECONDS,
        max_jump: float = config.STROKE_MAX_JUMP,
        resume_max_distance: float = config.RESUME_MAX_DISTANCE,
        pointer_min_cutoff: float = config.POINTER_MIN_CUTOFF_HZ,
        pointer_beta: float = config.POINTER_BETA,
        toolbar_min_cutoff: float = config.TOOLBAR_POINTER_MIN_CUTOFF_HZ,
        toolbar_beta: float = config.TOOLBAR_POINTER_BETA,
    ) -> None:
        self.canvas = canvas
        self.save_handler = save_handler
        self.toolbar = toolbar if toolbar is not None else build_toolbar()
        # UNKNOWN only replaces the current gesture after the grace period, so a
        # few uncertain frames pause drawing/dwell instead of cancelling them.
        self.stabilizer = GestureStabilizer(
            hold_seconds,
            {Gesture.SELECT: select_hold_seconds, Gesture.UNKNOWN: uncertain_grace_seconds},
            resume_seconds=config.DROPOUT_RESUME_SECONDS,
        )
        self.dwell_seconds = dwell_seconds
        self.action_dwell_seconds = action_dwell_seconds
        self.hover_max_speed = hover_max_speed
        self.dwell_grace_seconds = dwell_grace_seconds
        self.hover_margin = hover_margin
        self.cooldown_seconds = cooldown_seconds
        self.save_clear_cooldown_seconds = save_clear_cooldown_seconds
        self.clear_confirm_seconds = clear_confirm_seconds
        self.max_gap_seconds = max_gap_seconds
        self.max_jump = max_jump
        self.resume_max_distance = resume_max_distance
        self._filter = PointFilter(pointer_min_cutoff, pointer_beta)
        self._draw_filter_params = (pointer_min_cutoff, pointer_beta)
        self._toolbar_filter_params = (toolbar_min_cutoff, toolbar_beta)

        self.color_name = next(iter(config.BRUSH_COLORS))
        self.size_name = "Medium"
        canvas.set_color(hex_rgb(config.BRUSH_COLORS[self.color_name]))
        canvas.set_thickness(config.BRUSH_SIZES[self.size_name])

        # Public, read-only status for the UI and the diagnostics overlay.
        self.raw_gesture = Gesture.NO_HAND
        self.raw_reason = "no hand"
        self.raw_pointer: Optional[Point] = None  # Unfiltered fingertip.
        self.pointer: Optional[Point] = None  # Filtered fingertip (drawing / toolbar).
        self.pointer_speed = 0.0  # Filtered fingertip speed, normalized units per second.
        self.zone: Optional[Zone] = None
        self.hovered: Optional[ToolbarItem] = None
        self.dwell_progress = 0.0
        self.latched: Optional[ToolbarItem] = None
        self.status = ""  # Why nothing happened this frame, if something was held back.
        self.message = ""
        self.last_sequence = -1
        # Recent executed commands (diagnostics/tests). Bounded: a long session must not
        # grow memory without limit.
        self.executed: list[Command] = []
        self.recent: deque[tuple[Gesture, str]] = deque(maxlen=90)  # About 3 s of raw results.

        self._dwell_accum = 0.0
        self._dwell_last_t = 0.0
        self._dwell_miss_since: Optional[float] = None
        self._next_command_time = -math.inf
        self._last_save_time = -math.inf
        self._clear_armed_until = -math.inf
        self._stroke_owner: Optional[str] = None  # "gesture" | "mouse"
        self._last_draw_time = -math.inf
        self._pen_paused = False
        self._pending_draw: list[Point] = []
        self._last_time: Optional[float] = None

    # --- status helpers ------------------------------------------------------
    @property
    def gesture(self) -> Gesture:
        return self.stabilizer.stable

    def clear_armed(self, now: float) -> bool:
        return now <= self._clear_armed_until

    def dwell_for(self, item: ToolbarItem) -> float:
        if item.command.action in ACTIONS_NEEDING_LONG_DWELL:
            return self.action_dwell_seconds
        return self.dwell_seconds

    @property
    def interaction(self) -> str:
        """What the hand is doing right now, in words for the UI."""
        stable = self.stabilizer.stable
        if self.pointer is None:
            return "No hand"
        if stable in POINTING and self.zone is Zone.TOOLBAR:
            return "Selecting"
        if stable is Gesture.DRAW:
            return "Drawing" if self.zone is Zone.CANVAS else "Pen up"
        if stable is Gesture.SELECT:
            return "Pointing"
        if stable is Gesture.UNKNOWN:
            return "Unclear"
        if stable is Gesture.IDLE:
            return "Pen up"
        return "Ready"

    def recent_summary(self) -> str:
        """Share of raw gestures (and the top rejection reason) over the last ~3 s."""
        if not self.recent:
            return ""
        counts: dict[str, int] = {}
        reasons: dict[str, int] = {}
        for gesture, reason in self.recent:
            counts[gesture.value] = counts.get(gesture.value, 0) + 1
            if gesture is Gesture.UNKNOWN:
                reasons[reason] = reasons.get(reason, 0) + 1
        n = len(self.recent)
        text = " · ".join(f"{k} {100 * v // n}%" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        if reasons:
            top, cnt = max(reasons.items(), key=lambda kv: kv[1])
            text += f"\ntop reject: {top} ({100 * cnt // n}%)"
        return text

    # --- tracking input --------------------------------------------------------
    def process(
        self,
        sequence: int,
        timestamp: float,
        fingers: Optional[FingerCountResult],
        fingertip: Optional[Point],
    ) -> bool:
        """Handle one tracking snapshot. Returns False if it was already processed."""
        if sequence <= self.last_sequence:
            return False  # Duplicate or out-of-order snapshot: never act twice.
        self.last_sequence = sequence
        self.status = ""

        if fingertip is None:
            raw, reason = Gesture.NO_HAND, "no hand"
        else:
            raw, reason = explain_gesture(fingers)
        self.raw_gesture, self.raw_reason = raw, reason
        self.recent.append((raw, reason))
        previous = self.stabilizer.stable
        stable = self.stabilizer.update(raw, timestamp)
        self._update_pointer(fingertip, timestamp)
        pointer = self.pointer

        if pointer is not None and self.latched is not None and not self.latched.contains(pointer, self.hover_margin):
            self.latched = None  # Pointer left the item: it may be used again.

        zone = zone_of(pointer) if pointer is not None else None
        self.zone = zone

        # --- drawing (the toolbar zone and buffer intercept it first) ---
        if zone is Zone.TOOLBAR or (zone is Zone.BUFFER and not self._may_continue(pointer)):
            self._pending_draw.clear()
            if self._stroke_owner == "gesture":
                self._end_gesture_stroke()  # Entering the toolbar ends the stroke cleanly.
            if raw is Gesture.DRAW and stable is Gesture.DRAW:
                self.status = "toolbar: pen lifted" if zone is Zone.TOOLBAR else "move down to draw"
        elif raw is Gesture.DRAW and pointer is not None:
            if stable is Gesture.DRAW:
                if previous is not Gesture.DRAW:
                    # The frames that confirmed DRAW were real DRAW frames: draw
                    # them too, so the stroke starts where the gesture started.
                    for p in self._pending_draw:
                        self._draw_to(p, timestamp)
                self._pending_draw.clear()
                self._draw_to(pointer, timestamp)
            else:
                self._pending_draw.append(pointer)
                self.status = "confirming DRAW"
        else:
            self._pending_draw.clear()
            if stable is Gesture.DRAW and self._stroke_owner == "gesture":
                self._pen_paused = True  # Brief uncertainty: keep the stroke open.
                self.status = f"pen paused ({reason})"
        if stable is not Gesture.DRAW:
            self._end_gesture_stroke()

        # --- toolbar: one pointing finger hovers and selects ---
        if stable in POINTING and zone is Zone.TOOLBAR:
            if raw in POINTING:
                self._dwell_step(pointer, timestamp)
            else:
                self._dwell_pause(timestamp, f"dwell paused ({reason})")
        elif stable in POINTING and pointer is not None:
            self._dwell_pause(timestamp, "left the toolbar")  # Resets after the grace period.
        elif stable is Gesture.NO_HAND and raw in (Gesture.NO_HAND, *POINTING):
            # Hand not seen (or seen again, pointing, but not re-confirmed yet).
            # A drop-out of a frame or two only pauses the hover; if the hand
            # stays away longer than the dwell grace, the hover is cancelled.
            self._dwell_pause(timestamp, "tracking dropped")
        else:
            self._dwell_reset()  # Fist or prolonged uncertainty: cancel the hover.
        return True

    def _may_continue(self, point: Optional[Point]) -> bool:
        """A running gesture stroke may continue into the buffer down to DRAW_MIN_Y."""
        return point is not None and self._stroke_owner == "gesture" and point[1] >= DRAW_MIN_Y

    def _update_pointer(self, fingertip: Optional[Point], timestamp: float) -> None:
        previous_raw, previous_time = self.raw_pointer, self._last_time
        previous_filtered = self.pointer
        self.raw_pointer, self._last_time = fingertip, timestamp
        if fingertip is None:
            self._filter.reset()
            self.pointer = None
            self.pointer_speed = 0.0
            return
        if (
            previous_raw is None
            or previous_time is None
            or timestamp - previous_time > self.max_gap_seconds
            or math.dist(previous_raw, fingertip) > self.max_jump
        ):
            self._filter.reset()  # Never smooth across a gap or a jump.
            previous_filtered = None
        # Over the toolbar, steadiness matters more than latency: smooth harder.
        near_toolbar = previous_filtered is not None and previous_filtered[1] <= DRAW_START_Y
        self._filter.set_params(*(self._toolbar_filter_params if near_toolbar else self._draw_filter_params))
        self.pointer = self._filter(fingertip, timestamp)
        if previous_filtered is not None and previous_time is not None and timestamp > previous_time:
            self.pointer_speed = math.dist(previous_filtered, self.pointer) / (timestamp - previous_time)
        else:
            self.pointer_speed = 0.0

    def tracking_lost(self) -> None:
        """No fresh snapshots arrived: lift the pen immediately."""
        self.stabilizer.reset()
        self.raw_gesture, self.raw_reason = Gesture.NO_HAND, "no recent frames"
        self.raw_pointer = None
        self.pointer = None
        self._filter.reset()
        self._pending_draw.clear()
        self._dwell_reset()
        self._end_gesture_stroke()

    def reset_interaction(self) -> None:
        """Used on mode switches: finish any stroke and forget gesture state."""
        self.tracking_lost()
        if self._stroke_owner == "mouse":
            self.canvas.end_stroke()
            self._stroke_owner = None

    def _draw_to(self, point: Point, timestamp: float) -> None:
        if self._stroke_owner == "mouse":
            return  # Mouse is drawing; don't mix inputs into one stroke.
        if point[1] < DRAW_MIN_Y:
            self._end_gesture_stroke()  # Never draw underneath the toolbar.
            self.status = "toolbar band: no drawing"
            return
        if self._stroke_owner is None and point[1] < DRAW_START_Y:
            self.status = "move down to draw"  # New strokes start clear of the toolbar.
            return
        if self._stroke_owner == "gesture":
            active = self.canvas.active_stroke
            last = active.points[-1] if active else None
            distance = math.dist(last, point) if last is not None else 0.0
            gap = timestamp - self._last_draw_time > self.max_gap_seconds
            jump = distance > self.max_jump
            # After a pause, continue only if the finger barely moved. Otherwise a
            # straight line would cross the part that was not seen as DRAW.
            moved_while_paused = self._pen_paused and distance > self.resume_max_distance
            if gap or jump or moved_while_paused:
                self._end_gesture_stroke()
        self._pen_paused = False
        try:
            if self._stroke_owner is None:
                self.canvas.begin_stroke(point)
                self._stroke_owner = "gesture"
            else:
                self.canvas.add_point(point)
        except CanvasFullError as exc:
            self.message = str(exc)
            self._stroke_owner = None
            return
        self._last_draw_time = timestamp

    def _end_gesture_stroke(self) -> None:
        self._pen_paused = False
        if self._stroke_owner == "gesture":
            self.canvas.end_stroke()
            self._stroke_owner = None

    # --- dwell ------------------------------------------------------------------
    def _hit(self, point: Point) -> Optional[ToolbarItem]:
        # Hysteresis: the hovered item keeps a slightly larger hit box, so
        # jitter on its border does not flip between neighbouring items.
        if self.hovered is not None and self.hovered.contains(point, self.hover_margin):
            return self.hovered
        return next((i for i in self.toolbar if i.contains(point)), None)

    def _dwell_reset(self) -> None:
        self.hovered = None
        self.dwell_progress = 0.0
        self._dwell_accum = 0.0
        self._dwell_miss_since = None

    def _dwell_pause(self, now: float, why: str) -> None:
        """A miss (noisy frame, pointer slipped off): pause; reset only after a grace period."""
        if self.hovered is None:
            return
        if self._dwell_miss_since is None:
            self._dwell_miss_since = now
        self._dwell_last_t = now  # Paused time never counts towards the dwell.
        if now - self._dwell_miss_since > self.dwell_grace_seconds:
            self._dwell_reset()
        else:
            self.status = why

    def _dwell_step(self, point: Point, now: float) -> None:
        item = self._hit(point)
        if item is None:
            self._dwell_pause(now, "dwell paused (off the item)")
            return
        if item is self.latched:
            self.hovered, self.dwell_progress = item, 1.0
            self.status = "done · move off the item to use it again"
            return
        if item is not self.hovered:
            self.hovered, self._dwell_accum, self._dwell_last_t = item, 0.0, now
            self._dwell_miss_since = None
            self.dwell_progress = 0.0
            return
        dt = min(now - self._dwell_last_t, 0.1)  # Clamp: a stall cannot jump the timer.
        self._dwell_last_t = now
        self._dwell_miss_since = None
        if now < self._next_command_time:
            self._dwell_accum = 0.0
            self.dwell_progress = 0.0
            self.status = "cooldown"
            return
        if self.pointer_speed > self.hover_max_speed:
            self.status = "passing over · hold still to select"
            return  # Sweeping across the toolbar: the timer neither advances nor resets.
        self._dwell_accum += max(0.0, dt)
        self.dwell_progress = min(1.0, self._dwell_accum / self.dwell_for(item))
        if item.command.action is Action.CLEAR and self.clear_armed(now):
            self.status = "keep holding to confirm Clear"
        if self.dwell_progress >= 1.0:
            self.execute(item.command, now)
            if item.command.action is Action.CLEAR and self.clear_armed(now):
                # First step only armed Clear. Hold-to-confirm: keep resting on Clear
                # for another dwell to clear. No latch and no cooldown here, so the
                # user never has to move onto the neighbouring Undo/Save to confirm.
                self._dwell_accum = 0.0
                self.dwell_progress = 0.0
                self.message = "Clear armed · keep holding to confirm (Undo brings it back)"
                return
            self.latched = item
            extra = self.save_clear_cooldown_seconds if item.command.action in (Action.SAVE, Action.CLEAR) else 0.0
            self._next_command_time = now + self.cooldown_seconds + extra

    # --- mouse input (accessibility / testing) ---------------------------------
    def mouse_press(self, point: Point, now: float) -> None:
        item = next((i for i in self.toolbar if i.contains(point)), None)
        if item is not None:
            self.execute(item.command, now)
            return
        if self._stroke_owner is None and point[1] >= DRAW_MIN_Y:
            try:
                self.canvas.begin_stroke(point)
                self._stroke_owner = "mouse"
            except CanvasFullError as exc:
                self.message = str(exc)

    def mouse_drag(self, point: Point) -> None:
        if self._stroke_owner == "mouse":
            self.canvas.add_point((point[0], max(point[1], DRAW_MIN_Y)))

    def mouse_release(self) -> None:
        if self._stroke_owner == "mouse":
            self.canvas.end_stroke()
            self._stroke_owner = None

    # --- commands ----------------------------------------------------------------
    def execute(self, command: Command, now: float) -> str:
        """Run a toolbar/keyboard command and return a status message."""
        action = command.action
        if action is not Action.CLEAR:
            self._clear_armed_until = -math.inf
        if action is Action.COLOR:
            self.color_name = command.value
            self.canvas.set_color(hex_rgb(config.BRUSH_COLORS[command.value]))
            msg = f"Brush · {command.value}"
        elif action is Action.SIZE:
            self.size_name = command.value
            self.canvas.set_thickness(config.BRUSH_SIZES[command.value])
            msg = f"Size · {command.value}"
        elif action is Action.CYCLE_SIZE:
            names = list(config.BRUSH_SIZES)
            return self.execute(Command(Action.SIZE, names[(names.index(self.size_name) + 1) % len(names)]), now)
        elif action is Action.ERASER:
            eraser_on = self.canvas.tool is not Tool.ERASER
            self.canvas.set_tool(Tool.ERASER if eraser_on else Tool.BRUSH)
            msg = "Eraser on" if eraser_on else f"Brush · {self.color_name}"
        elif action is Action.UNDO:
            msg = "Undone" if self.canvas.undo() else "Nothing to undo"
            self._stroke_owner = None
        elif action is Action.CLEAR:
            msg = self._clear(now)
        elif action is Action.SAVE:
            msg = self._save(now)
        else:  # pragma: no cover - exhaustive enum
            msg = ""
        self.executed.append(command)
        if len(self.executed) > EXECUTED_LOG_LIMIT:
            del self.executed[: len(self.executed) - EXECUTED_LOG_LIMIT]
        self.message = msg
        return msg

    def _clear(self, now: float) -> str:
        if self.canvas.is_empty:
            self._clear_armed_until = -math.inf
            return "Canvas is already empty"
        if now <= self._clear_armed_until:
            self._clear_armed_until = -math.inf
            self._stroke_owner = None
            self.canvas.clear()
            return "Canvas cleared · Undo brings it back"
        self._clear_armed_until = now + self.clear_confirm_seconds
        return f"Select Clear again within {self.clear_confirm_seconds:.0f} s to confirm"

    def _save(self, now: float) -> str:
        if self.canvas.is_empty:
            return "Nothing to save yet"
        if now - self._last_save_time < self.save_clear_cooldown_seconds:
            return "Already saved"
        if self.save_handler is None:
            return "Saving is not available"
        self._last_save_time = now
        try:
            path = self.save_handler()
        except Exception as exc:
            return f"Save failed: {exc}"
        return f"Saved {path.name}"
