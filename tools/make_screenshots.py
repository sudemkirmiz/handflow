"""Render README screenshots from synthetic data (Windows only).

    python -m tools.make_screenshots [output_dir] [WIDTHxHEIGHT]

No camera is used: the "camera" is a plain frame, the hand is a synthetic
skeleton, and strokes are drawn programmatically. The window is invisible,
click-through and never activated (see ``tools.offscreen_capture``), so this
does not disturb the desktop.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.air_canvas import Action, Command  # noqa: E402
from core.camera import CameraState  # noqa: E402
from core.finger_counter import Finger, count_fingers  # noqa: E402
from core.hand_tracker import Landmark, TrackedHand  # noqa: E402
from core.overlay import draw_hand  # noqa: E402
from core.processing import FrameSnapshot, TrackingState  # noqa: E402
from tests.synthetic_hand import make_hand, make_posed_hand  # noqa: E402
from tools.offscreen_capture import capture, foreground_is_ours, make_ghost, toplevel_hwnd  # noqa: E402
from ui.app import HandFlowApp, Mode  # noqa: E402

FRAME_SHAPE = (480, 640)


class StaticSource:
    """Serves prepared snapshots; no camera, no worker thread."""

    def __init__(self) -> None:
        self.history: list[FrameSnapshot] = []

    def start(self) -> None: ...
    def stop(self, timeout: float = 3.0) -> None: ...
    def set_draw_landmarks(self, enabled: bool) -> None: ...

    def latest(self):
        return self.history[-1] if self.history else None

    def snapshots_since(self, sequence: int):
        return [s for s in self.history if s.sequence > sequence]


def synthetic_frame() -> np.ndarray:
    """A neutral studio-grey frame with a soft vertical falloff (no camera image)."""
    h, w = FRAME_SHAPE
    ramp = np.linspace(46, 30, h, dtype=np.float32)[:, None]
    frame = np.repeat(np.repeat(ramp, w, axis=1)[:, :, None], 3, axis=2)
    frame[..., 2] += 4  # Slightly cool.
    return np.clip(frame, 0, 255).astype(np.uint8)


def image_hand(world: np.ndarray, center=(0.5, 0.62), scale=2.6, tip=None) -> TrackedHand:
    """Project a synthetic 3D hand into normalized image coordinates."""
    pts = [Landmark(center[0] + p[0] * scale, center[1] - p[1] * scale, p[2]) for p in world]
    if tip is not None:
        dx, dy = tip[0] - pts[8].x, tip[1] - pts[8].y
        pts = [Landmark(p.x + dx, p.y + dy, p.z) for p in pts]
    return TrackedHand(tuple(pts), tuple(Landmark(*p) for p in world), "Right", 0.97)


def snapshot(seq: int, t: float, frame: np.ndarray, hand: TrackedHand, fingers) -> FrameSnapshot:
    return FrameSnapshot(seq, t, CameraState.CONNECTED, "Connected · 640×480", TrackingState.TRACKING,
                         frame, hand, fingers, 30.0)


def settle(app: HandFlowApp, seconds: float = 0.4) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.update()
        time.sleep(0.01)


def new_app(geometry: str) -> HandFlowApp:
    HandFlowApp._deactivate_windows_window_header_manipulation = True  # No show/hide flash.
    app = HandFlowApp(StaticSource())
    make_ghost(app)
    app.geometry(geometry + "+0+0")
    return app


def shoot(app: HandFlowApp, path: Path) -> None:
    settle(app)
    capture(toplevel_hwnd(app), app.winfo_width(), app.winfo_height()).save(path)


def hand_tracking(path: Path, geometry: str) -> None:
    app = new_app(geometry)
    world = make_posed_hand({Finger.RING: (70, 100, 70), Finger.PINKY: (70, 100, 70)}, (0, 0), 45)
    hand = image_hand(world)
    frame = synthetic_frame()
    draw_hand(frame, hand)
    app._source.history.append(snapshot(1, time.monotonic(), frame, hand, count_fingers(world)))
    app.process_new_snapshots()
    shoot(app, path)
    app.shutdown()


def air_canvas(path: Path, geometry: str) -> None:
    app = new_app(geometry)
    app.set_mode(Mode.AIR_CANVAS)
    settle(app, 0.2)
    ctl = app.controller

    def stroke(points):
        ctl.mouse_press(points[0], 0)
        for p in points[1:]:
            ctl.mouse_drag(p)
        ctl.mouse_release()

    ctl.execute(Command(Action.COLOR, "Purple"), 0)
    heart = [(0.5 + 0.15 * math.sin(t) ** 3,
              0.56 - 0.011 * (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)))
             for t in np.linspace(0, 2 * math.pi, 140)]
    stroke(heart)
    ctl.execute(Command(Action.COLOR, "Mint"), 0)
    ctl.execute(Command(Action.SIZE, "Thick"), 0)
    stroke([(0.14 + 0.72 * k / 90, 0.86 + 0.035 * math.sin(k / 7)) for k in range(91)])
    ctl.execute(Command(Action.COLOR, "Yellow"), 0)
    ctl.execute(Command(Action.SIZE, "Thin"), 0)
    for x, y in [(0.2, 0.38), (0.8, 0.4), (0.24, 0.66), (0.77, 0.68)]:
        stroke([(x - 0.025, y), (x + 0.025, y)])
        stroke([(x, y - 0.035), (x, y + 0.035)])
    ctl.execute(Command(Action.COLOR, "Blue"), 0)
    ctl.execute(Command(Action.SIZE, "Medium"), 0)
    ctl.message = ""

    # One index finger resting on "Save": about two thirds of the dwell done.
    save = next(i for i in ctl.toolbar if i.command.action is Action.SAVE)
    target = ((save.rect[0] + save.rect[2]) / 2, (save.rect[1] + save.rect[3]) / 2)
    world = make_posed_hand({f: (70, 100, 70) for f in (Finger.MIDDLE, Finger.RING, Finger.PINKY)}, (35, 45), 10)
    fingers = count_fingers(world)
    hand = image_hand(world, tip=target)
    frame = synthetic_frame()
    t0 = time.monotonic() - 1.0
    for k in range(1, 26):
        app._source.history.append(snapshot(k, t0 + k / 30, frame, hand, fingers))
    app.process_new_snapshots()
    shoot(app, path)
    app.shutdown()


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "screenshots"
    geometry = sys.argv[2] if len(sys.argv) > 2 else "1180x720"
    only = sys.argv[3] if len(sys.argv) > 3 else None
    out.mkdir(parents=True, exist_ok=True)
    if only is None:
        # One process per window: CustomTkinter keeps per-window timers that
        # complain when a second window is created in the same process.
        import subprocess

        for name in ("air_canvas", "hand_tracking"):
            subprocess.run([sys.executable, "-m", "tools.make_screenshots", str(out), geometry, name],
                           cwd=ROOT, check=True)
        return
    {"air_canvas": air_canvas, "hand_tracking": hand_tracking}[only](out / f"{only}.png", geometry)
    print(f"{only}.png saved · focus held by this process at the end: {foreground_is_ours()}")


if __name__ == "__main__":
    main()
