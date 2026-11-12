"""Central configuration for HandFlow: paths, camera, tracking thresholds, theme."""

from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
MODEL_PATH: Path = PROJECT_ROOT / "models" / "hand_landmarker.task"
# Official MediaPipe model (float16, version 1). See models/README.md.
MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256: str = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"

# --- Camera ------------------------------------------------------------------
CAMERA_INDEX: int = 0
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480
MIRROR_PREVIEW: bool = True
# After this many consecutive failed reads the camera is treated as lost.
MAX_CONSECUTIVE_READ_FAILURES: int = 15
# Seconds to wait between reconnection attempts.
CAMERA_RETRY_SECONDS: float = 2.0

# --- Hand tracking -------------------------------------------------------------
MAX_HANDS: int = 1  # Stage 1 tracks one hand; the pipeline already handles lists.
MIN_HAND_DETECTION_CONFIDENCE: float = 0.6
MIN_HAND_PRESENCE_CONFIDENCE: float = 0.5
MIN_TRACKING_CONFIDENCE: float = 0.5
# Finger analysis over time (core.finger_counter.FingerTracker). Chosen by
# simulation with MediaPipe-like landmark noise: a relaxed open hand reads 5 in
# ~99% of frames (3.9% per-frame), fists stay 100%, open->fist in ~1 frame.
FINGER_SMOOTHING: float = 0.35  # EMA weight of the newest landmarks.
FINGER_HYSTERESIS_BEND: float = 15.0  # Degrees a finger must clearly pass to flip.
FINGER_HYSTERESIS_REACH: float = 0.07

# --- UI ----------------------------------------------------------------------
UI_POLL_MS: int = 15  # How often the UI checks for a new processed frame.
# If no new frame arrives for this long, hand info on screen is cleared.
STALE_RESULT_SECONDS: float = 0.5
WINDOW_TITLE: str = "HandFlow Studio"
WINDOW_SIZE: str = "1180x720"
WINDOW_MIN_SIZE: tuple[int, int] = (960, 600)

# --- Air Canvas ------------------------------------------------------------------
OUTPUT_DIR: Path = PROJECT_ROOT / "output"  # PNG exports (drawing only, never camera images).
EXPORT_LONG_SIDE: int = 1600  # Exported PNG long side in pixels; aspect follows the camera.
BRUSH_COLORS: dict[str, str] = {
    "Pink": "#FF8FCF",
    "Purple": "#B58CFF",
    "Blue": "#7FB8FF",
    "Mint": "#8EF0C4",
    "Yellow": "#FFE38F",
    "White": "#FFFFFF",
}
# Stroke thickness as a fraction of the canvas height (resolution independent).
BRUSH_SIZES: dict[str, float] = {"Thin": 0.008, "Medium": 0.016, "Thick": 0.030}
ERASER_SIZE_FACTOR: float = 2.5  # Eraser is this many times wider than the brush.
# PNG export backgrounds. "Dark" matches the app background (COLORS["background"]).
EXPORT_BACKGROUNDS: dict[str, str | None] = {"Transparent": None, "White": "#FFFFFF", "Dark": "#111318"}

MAX_STROKES: int = 2000  # Hard cap on stored strokes (memory stays bounded).
MAX_POINTS_PER_STROKE: int = 4000  # Longer strokes are continued as a new stroke.
UNDO_LIMIT: int = 50  # How many actions can be undone.
MIN_POINT_DISTANCE: float = 0.002  # Normalized; skip points closer than this.

# Gesture timing (seconds) and pointer filtering.
GESTURE_HOLD_SECONDS: float = 0.12  # A new gesture must persist this long.
SELECT_HOLD_SECONDS: float = 0.18
# One-finger toolbar: the pointing finger selects by resting on an item.
DWELL_SECONDS: float = 0.5  # Colors, size, eraser (harmless, reversible).
ACTION_DWELL_SECONDS: float = 1.0  # Undo, Clear, Save need a longer, deliberate rest.
# Faster than this (normalized frame widths / s) the finger is passing over the
# toolbar, not resting on it, so the dwell timer does not advance.
HOVER_MAX_SPEED: float = 0.6
COMMAND_COOLDOWN_SECONDS: float = 0.6  # Between any two toolbar commands.
SAVE_CLEAR_COOLDOWN_SECONDS: float = 2.0  # Extra cooldown for Save / Clear.
CLEAR_CONFIRM_SECONDS: float = 6.0  # Window for the second, confirming Clear.
STROKE_MAX_GAP_SECONDS: float = 0.5  # Longer gaps between snapshots start a new stroke.
STROKE_MAX_JUMP: float = 0.15  # Normalized jump that starts a new stroke.
# Brief classification uncertainty (hand still tracked, gesture UNKNOWN) keeps
# the current gesture this long before it counts as a real change. During it
# the pen pauses (no points are added) and the dwell timer pauses.
UNCERTAIN_GRACE_SECONDS: float = 0.35
# Tracking drop-outs (the hand vanishes for a frame or two, typical near the
# image edges): the pen still lifts at once, but if the same gesture is back
# within this time it continues without re-confirming, and a toolbar hover
# only pauses (up to DWELL_GRACE_SECONDS) instead of restarting.
DROPOUT_RESUME_SECONDS: float = 0.15
# After a pause, the stroke continues only if the fingertip moved less than
# this (normalized); otherwise a new stroke starts, so no long line is drawn.
RESUME_MAX_DISTANCE: float = 0.05
# Toolbar dwell: brief misses (noisy frame, pointer slipping off) up to this
# long pause the timer instead of restarting it.
DWELL_GRACE_SECONDS: float = 0.3
HOVER_MARGIN: float = 0.02  # The hovered item's hit box grows by this (hysteresis).
# One Euro filter for the fingertip (normalized units). Chosen by simulation at
# 30 FPS: stationary jitter x0.35, lag 0.011 at 1 frame-width/s (EMA 0.5 was
# x0.55 and 0.033).
POINTER_MIN_CUTOFF_HZ: float = 1.0
POINTER_BETA: float = 10.0
# Stronger smoothing while the fingertip is over the toolbar (simulated: jitter
# x0.26 instead of x0.39; lag 0.021 at 0.5 frame-widths/s, fine for hovering).
TOOLBAR_POINTER_MIN_CUTOFF_HZ: float = 0.5
TOOLBAR_POINTER_BETA: float = 3.0
DIAGNOSTICS_DEFAULT: bool = False  # Air Canvas diagnostics overlay (D key / inspector switch).

# --- Theme (HandFlow Studio) ---------------------------------------------------------
COLORS: dict[str, str] = {
    # Accent is reserved for selection, focus and progress.
    "background": "#111318",
    "surface": "#1B1F27",
    "elevated": "#292F3B",
    "well": "#0C0E12",  # Behind the camera image (letterbox area).
    "blank_canvas": "#161A21",  # Air Canvas without a camera.
    "border": "#262B35",  # Derived neutral: hairline borders and dividers.
    "border_strong": "#363D4A",  # Derived neutral: hover outlines.
    "accent": "#8B8CF8",
    "accent_hover": "#A0A1FA",
    "accent_muted": "#30325A",  # Selected-state fill (accent at low contrast).
    "text": "#F4F5F8",
    "text_secondary": "#9CA3AF",
    "text_muted": "#6B7280",
    "success": "#6EE7B7",
    "warning": "#F3C77B",
    "error": "#F28B93",
    "on_accent": "#111318",
}


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' to an OpenCV BGR tuple."""
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' to an RGB tuple (frames are RGB inside HandFlow)."""
    b, g, r = hex_to_bgr(hex_color)
    return (r, g, b)
