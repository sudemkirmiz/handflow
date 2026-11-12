"""HandFlow Studio window: header, camera workspace and inspector, two modes.

The UI never touches the camera or the model directly. It reads immutable
``FrameSnapshot`` objects from the ``FrameProcessor`` in sequence order, feeds
them to the Air Canvas controller (each exactly once) and renders the newest.
"""

from __future__ import annotations

import time
import tkinter as tk
from enum import Enum
from typing import Optional, Protocol

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

import config
from core.air_canvas import HINT, Action, AirCanvasController, Command, ToolbarItem, Zone
from core.camera import CameraState
from core.canvas import DrawingCanvas, Tool
from core.canvas_render import LayerCache, composite, save_png
from core.finger_counter import Finger, FingerState
from core.gestures import Gesture
from core.processing import FrameSnapshot, TrackingState
from core.viewport import Viewport, fit_viewport
from ui import theme
from ui.canvas_panel import AirCanvasPanel
from ui.theme import C

INDEX_TIP = 8
HAND_TRACKING_HINT = "Show one hand to the camera, about 40–80 cm away"


class SnapshotSource(Protocol):
    def start(self) -> None: ...
    def stop(self, timeout: float = ...) -> None: ...
    def latest(self) -> Optional[FrameSnapshot]: ...
    def snapshots_since(self, sequence: int) -> list[FrameSnapshot]: ...
    def set_draw_landmarks(self, enabled: bool) -> None: ...


class Mode(Enum):
    HAND_TRACKING = "Hand Tracking"
    AIR_CANVAS = "Air Canvas"


_CAMERA_DOT = {
    CameraState.CONNECTED: C["success"],
    CameraState.DISCONNECTED: C["warning"],
    CameraState.UNAVAILABLE: C["error"],
    CameraState.READ_FAILED: C["error"],
}
_TRACKING_COLOR = {
    TrackingState.TRACKING: C["success"],
    TrackingState.SEARCHING: C["text_secondary"],
    TrackingState.WAITING_FOR_CAMERA: C["warning"],
    TrackingState.MODEL_UNAVAILABLE: C["error"],
    TrackingState.ERROR: C["error"],
}
_FINGER_STYLE = {  # state -> (text, fill, text color)
    FingerState.EXTENDED: ("Up", C["accent_muted"], C["accent"]),
    FingerState.FOLDED: ("Down", C["elevated"], C["text_muted"]),
    FingerState.UNCERTAIN: ("?", C["elevated"], C["warning"]),
}
_BLANK_CANVAS = (config.CAMERA_HEIGHT, config.CAMERA_WIDTH)  # Used for mouse drawing without a camera.


def _camera_pill_text(snap: FrameSnapshot) -> str:
    """Short header text; the full explanation is shown in the preview area."""
    if snap.camera_state is CameraState.CONNECTED:
        return snap.camera_message
    return {
        CameraState.DISCONNECTED: "Connecting…",
        CameraState.UNAVAILABLE: "Camera unavailable",
        CameraState.READ_FAILED: "Camera lost · reconnecting",
    }[snap.camera_state]


def fingertip(snap: FrameSnapshot) -> Optional[tuple[float, float]]:
    """Index fingertip in normalized coordinates of the displayed (mirrored) frame."""
    if snap.hand is None:
        return None
    tip = snap.hand.landmarks[INDEX_TIP]
    return (tip.x, tip.y)


class HandFlowApp(ctk.CTk):
    def __init__(self, source: SnapshotSource, controller: Optional[AirCanvasController] = None) -> None:
        super().__init__(fg_color=C["background"])
        self._source = source
        self._poll_job: Optional[str] = None
        self._closing = False
        self._last_sequence = -1
        self._last_snapshot: Optional[FrameSnapshot] = None
        self._showing_hand = False
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._last_frame: Optional[np.ndarray] = None
        self._viewport: Optional[Viewport] = None
        self._layer_cache = LayerCache()
        self.mode = Mode.HAND_TRACKING
        self.frames_rendered = 0
        self.diagnostics = config.DIAGNOSTICS_DEFAULT
        self._diag_items: Optional[tuple[int, int]] = None
        self._image_visible = False
        self._configured: dict[tuple[int, str], object] = {}
        self._toolbar_signature: Optional[tuple] = None
        self._cursor_items: Optional[tuple[int, int]] = None

        canvas = DrawingCanvas(
            eraser_factor=config.ERASER_SIZE_FACTOR,
            max_strokes=config.MAX_STROKES,
            max_points_per_stroke=config.MAX_POINTS_PER_STROKE,
            undo_limit=config.UNDO_LIMIT,
            min_point_distance=config.MIN_POINT_DISTANCE,
        )
        self.controller = controller or AirCanvasController(canvas, save_handler=self._save_drawing)
        if self.controller.save_handler is None:
            self.controller.save_handler = self._save_drawing

        self.title(config.WINDOW_TITLE)
        self._fit_to_screen()
        self.minsize(*config.WINDOW_MIN_SIZE)
        self.protocol("WM_DELETE_WINDOW", self.shutdown)

        self.fonts = theme.Fonts(self)  # Created once, shared by all widgets.
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self._build_header()
        self._build_main()
        self._bind_keys()
        self._apply_mode_styles()

    # --- layout ----------------------------------------------------------------
    def _fit_to_screen(self) -> None:
        """Use the preferred size, shrunk to fit small or high-DPI screens, centered."""
        scaling = ctk.ScalingTracker.get_window_scaling(self)
        screen_w = self.winfo_screenwidth() / scaling
        screen_h = self.winfo_screenheight() / scaling
        pref_w, pref_h = (int(v) for v in config.WINDOW_SIZE.split("x"))
        w = int(min(pref_w, screen_w - 40))
        h = int(min(pref_h, screen_h - 90))  # Leave room for the taskbar and title bar.
        x = int(max(0, (screen_w - w) / 2))
        y = int(max(0, (screen_h - h) / 2 - 20))
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _build_header(self) -> None:
        f = self.fonts
        header = ctk.CTkFrame(self, fg_color=C["background"], corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="w", padx=(theme.SPACE_LG, theme.SPACE_XL), pady=theme.SPACE_MD)
        ctk.CTkFrame(brand, width=12, height=12, corner_radius=3, fg_color=C["accent"]).pack(
            side="left", padx=(0, theme.SPACE_SM))
        ctk.CTkLabel(brand, text="HandFlow Studio", font=f.brand, text_color=C["text"]).pack(side="left")

        # Mode switch: a quiet segmented control (selection = raised surface, no accent fill).
        switch = ctk.CTkFrame(header, fg_color=C["surface"], corner_radius=theme.RADIUS_MD, border_width=1,
                              border_color=C["border"])
        switch.grid(row=0, column=1, sticky="w")
        self._mode_buttons: dict[Mode, ctk.CTkButton] = {}
        for mode in Mode:
            btn = ctk.CTkButton(switch, text=mode.value, font=f.body_strong, width=118,
                                height=theme.CONTROL_SM - 4, corner_radius=theme.RADIUS_SM,
                                command=lambda m=mode: self.set_mode(m))
            btn.pack(side="left", padx=2, pady=2)
            self._mode_buttons[mode] = btn

        status = ctk.CTkFrame(header, fg_color="transparent")
        status.grid(row=0, column=2, sticky="e", padx=theme.SPACE_LG)
        self._camera_dot = ctk.CTkLabel(status, text="●", font=f.caption, text_color=C["warning"], width=12)
        self._camera_dot.pack(side="left", padx=(0, theme.SPACE_XS))
        self._camera_label = ctk.CTkLabel(status, text="Starting camera…", font=f.body,
                                          text_color=C["text_secondary"])
        self._camera_label.pack(side="left")
        theme.divider(self).grid(row=1, column=0, sticky="ew")

    def _build_main(self) -> None:
        f = self.fonts
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=2, column=0, sticky="nsew", padx=theme.SPACE_LG, pady=theme.SPACE_LG)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=0, minsize=theme.INSPECTOR_WIDTH)
        main.grid_rowconfigure(0, weight=1)

        # Workspace: the camera preview is the primary element.
        workspace = theme.panel(main)
        workspace.grid(row=0, column=0, sticky="nsew", padx=(0, theme.SPACE_LG))
        workspace.grid_rowconfigure(0, weight=1)
        workspace.grid_columnconfigure(0, weight=1)
        # A plain tk.Canvas gives exact pixel coordinates for mouse input and overlays.
        self._preview = tk.Canvas(workspace, bg=C["well"], highlightthickness=0, bd=0, width=200, height=150)
        self._preview.grid(row=0, column=0, sticky="nsew", padx=theme.SPACE_SM, pady=(theme.SPACE_SM, 0))
        self._image_item = self._preview.create_image(0, 0, anchor="nw")
        self._placeholder = self._preview.create_text(
            0, 0, text="Waiting for camera…", fill=C["text_secondary"], font=(theme.ui_family(self), 12),
            width=420, justify="center",
        )
        self._preview.bind("<Configure>", self._on_preview_resize, add="+")
        self._preview.bind("<ButtonPress-1>", self._on_mouse_press)
        self._preview.bind("<B1-Motion>", self._on_mouse_drag)
        self._preview.bind("<ButtonRelease-1>", self._on_mouse_release)

        footer = ctk.CTkFrame(workspace, fg_color="transparent")
        footer.grid(row=1, column=0, sticky="ew", padx=theme.SPACE_MD, pady=theme.SPACE_XS)
        self._hint_label = ctk.CTkLabel(footer, text=HAND_TRACKING_HINT, font=f.caption,
                                        text_color=C["text_secondary"], anchor="w", height=24)
        self._hint_label.pack(side="left")
        self._fps_value = ctk.CTkLabel(footer, text="— FPS", font=f.caption, text_color=C["text_muted"],
                                       height=24)
        self._fps_value.pack(side="right")

        # Inspector: fixed width, one panel per mode.
        self._sidebar = ctk.CTkFrame(main, fg_color="transparent", width=theme.INSPECTOR_WIDTH)
        self._sidebar.grid(row=0, column=1, sticky="nsew")
        self._sidebar.pack_propagate(False)
        self._analysis_card = theme.panel(self._sidebar)
        self._build_analysis_card(self._analysis_card)
        self._canvas_panel = AirCanvasPanel(self._sidebar, self.fonts, self.run_command, self.set_diagnostics)
        self._analysis_card.pack(fill="both", expand=True)

    def _build_analysis_card(self, card: ctk.CTkFrame) -> None:
        f, pad = self.fonts, {"padx": theme.SPACE_LG}

        # Model output (what MediaPipe reports) ...
        theme.section_label(card, "Model output", f).pack(fill="x", pady=(theme.SPACE_LG, theme.SPACE_XS), **pad)
        status_row = self._info_row_frame(card, "Status")
        self._tracking_label = ctk.CTkLabel(status_row, text="Waiting for camera", font=f.body_strong,
                                            text_color=C["warning"], height=24)
        self._tracking_label.pack(side="right")
        self._tracking_dot = ctk.CTkLabel(status_row, text="●", font=f.caption, text_color=C["warning"],
                                          width=12, height=24)
        self._tracking_dot.pack(side="right", padx=(0, theme.SPACE_XS))
        self._hand_value = self._info_row(card, "Hand")
        self._landmarks_value = self._info_row(card, "Landmarks")

        theme.divider(card).pack(fill="x", pady=theme.SPACE_MD, **pad)

        # ... and what HandFlow derives from it.
        theme.section_label(card, "Fingers (derived)", f).pack(fill="x", pady=(0, theme.SPACE_XS), **pad)
        metric = ctk.CTkFrame(card, fg_color="transparent")
        metric.pack(fill="x", **pad)
        self._count_label = ctk.CTkLabel(metric, text="—", font=f.metric, text_color=C["text_muted"], height=40)
        self._count_label.pack(side="left")
        ctk.CTkLabel(metric, text=" / 5 up", font=f.body, text_color=C["text_muted"], height=40).pack(
            side="left", pady=(8, 0))
        self._uncertain_label = ctk.CTkLabel(metric, text="", font=f.caption, text_color=C["warning"], height=40)
        self._uncertain_label.pack(side="right", pady=(8, 0))

        chips = ctk.CTkFrame(card, fg_color="transparent")
        chips.pack(fill="x", pady=(theme.SPACE_SM, 0), **pad)
        self._finger_chips: dict[Finger, ctk.CTkLabel] = {}
        for col, finger in enumerate(Finger):
            chips.grid_columnconfigure(col, weight=1, uniform="chips")
            ctk.CTkLabel(chips, text=finger.value, font=f.caption, text_color=C["text_muted"], height=18).grid(
                row=0, column=col)
            chip = ctk.CTkLabel(chips, text="—", width=44, height=24, corner_radius=theme.RADIUS_SM,
                                fg_color=C["elevated"], text_color=C["text_muted"], font=f.caption_strong)
            chip.grid(row=1, column=col, pady=(2, 0))
            self._finger_chips[finger] = chip

        self._detail_label = ctk.CTkLabel(card, text="", font=f.caption, text_color=C["text_secondary"],
                                          wraplength=theme.INSPECTOR_WIDTH - 2 * theme.SPACE_LG, justify="left",
                                          anchor="w")
        self._detail_label.pack(fill="x", pady=(theme.SPACE_MD, 0), **pad)

    def _info_row_frame(self, parent: ctk.CTkFrame, name: str) -> ctk.CTkFrame:
        row = ctk.CTkFrame(parent, fg_color="transparent", height=24)
        row.pack(fill="x", padx=theme.SPACE_LG)
        ctk.CTkLabel(row, text=name, font=self.fonts.body, text_color=C["text_secondary"], height=24).pack(
            side="left")
        return row

    def _info_row(self, parent: ctk.CTkFrame, name: str) -> ctk.CTkLabel:
        row = self._info_row_frame(parent, name)
        value = ctk.CTkLabel(row, text="—", font=self.fonts.body_strong, text_color=C["text"], height=24)
        value.pack(side="right")
        return value

    def _bind_keys(self) -> None:
        self.bind("<F1>", lambda _e: self.set_mode(Mode.HAND_TRACKING))
        self.bind("<F2>", lambda _e: self.set_mode(Mode.AIR_CANVAS))
        self.bind("<Key-d>", lambda _e: self.set_diagnostics(not self.diagnostics))
        keymap = {f"<Key-{i + 1}>": Command(Action.COLOR, name) for i, name in enumerate(config.BRUSH_COLORS)}
        keymap.update({
            "<Key-q>": Command(Action.SIZE, "Thin"),
            "<Key-w>": Command(Action.SIZE, "Medium"),
            "<Key-e>": Command(Action.SIZE, "Thick"),
            "<Key-x>": Command(Action.ERASER),
            "<Control-z>": Command(Action.UNDO),
            "<Control-s>": Command(Action.SAVE),
            "<Delete>": Command(Action.CLEAR),
        })
        self.keymap = keymap  # Exposed so tests can verify bindings without stealing OS focus.
        for sequence, command in keymap.items():
            self.bind(sequence, lambda _e, c=command: self._on_key_command(c))

    # --- modes -------------------------------------------------------------------
    def set_mode(self, mode: Mode) -> None:
        if mode is self.mode or self._closing:
            return
        self.controller.reset_interaction()  # Never carry a half-finished stroke across modes.
        self.mode = mode
        air = mode is Mode.AIR_CANVAS
        # Air Canvas shows a fingertip cursor instead of the full skeleton.
        self._source.set_draw_landmarks(not air)
        if air:
            self._analysis_card.pack_forget()
            self._canvas_panel.pack(fill="both", expand=True)
            self._update_canvas_panel()  # Populate it even before the first snapshot.
        else:
            self._canvas_panel.pack_forget()
            self._analysis_card.pack(fill="both", expand=True)
            self._clear_overlays()
        self._apply_mode_styles()
        self._redraw_last_frame()

    def _apply_mode_styles(self) -> None:
        for mode, btn in self._mode_buttons.items():
            active = mode is self.mode
            btn.configure(
                fg_color=C["elevated"] if active else "transparent",
                hover_color=C["elevated"] if active else C["border"],
                text_color=C["text"] if active else C["text_secondary"],
            )
        self._hint_label.configure(text=HINT if self.mode is Mode.AIR_CANVAS else HAND_TRACKING_HINT)

    # --- commands and input ---------------------------------------------------------
    def run_command(self, command: Command) -> str:
        message = self.controller.execute(command, time.monotonic())
        self._after_canvas_change()
        return message

    def _on_key_command(self, command: Command) -> None:
        if self.mode is Mode.AIR_CANVAS:
            self.run_command(command)

    def _event_point(self, event: tk.Event) -> Optional[tuple[float, float]]:
        if self.mode is not Mode.AIR_CANVAS or self._viewport is None:
            return None
        return self._viewport.to_normalized(event.x, event.y)

    def _on_mouse_press(self, event: tk.Event) -> None:
        point = self._event_point(event)
        if point is not None:
            self.controller.mouse_press(point, time.monotonic())
            self._after_canvas_change()

    def _on_mouse_drag(self, event: tk.Event) -> None:
        point = self._event_point(event)
        if point is not None:
            self.controller.mouse_drag(point)
            self._after_canvas_change()

    def _on_mouse_release(self, _event: tk.Event) -> None:
        self.controller.mouse_release()
        self._after_canvas_change()

    def _after_canvas_change(self) -> None:
        if self.mode is Mode.AIR_CANVAS:
            self._redraw_last_frame()
            self._update_canvas_panel()

    def _save_drawing(self):
        frame = self._last_frame
        h, w = frame.shape[:2] if frame is not None else _BLANK_CANVAS
        background = config.EXPORT_BACKGROUNDS[self._canvas_panel.background_name]
        return save_png(
            self.controller.canvas,
            config.OUTPUT_DIR,
            aspect=(w, h),
            long_side=config.EXPORT_LONG_SIDE,
            background=config.hex_to_rgb(background) if background else None,
        )

    # --- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        self._source.start()
        self._schedule_poll()

    def _schedule_poll(self) -> None:
        if not self._closing:
            self._poll_job = self.after(config.UI_POLL_MS, self._poll)

    def _poll(self) -> None:
        self._poll_job = None
        if self._closing:
            return
        self.process_new_snapshots()
        self._schedule_poll()

    def process_new_snapshots(self) -> int:
        """Feed every unseen snapshot to the controller in order; render the newest."""
        snapshots = self._source.snapshots_since(self._last_sequence)
        if not snapshots:
            latest = self._source.latest()
            if latest is not None and latest.sequence > self._last_sequence:
                snapshots = [latest]  # History overflowed; at least use the newest.
        if snapshots:
            for snap in snapshots:
                if self.mode is Mode.AIR_CANVAS:
                    self.controller.process(snap.sequence, snap.created_at, snap.fingers, fingertip(snap))
            self._last_sequence = snapshots[-1].sequence
            self._last_snapshot = snapshots[-1]
            self.render(snapshots[-1])
        else:
            last = self._source.latest()
            if last is not None and time.monotonic() - last.created_at > config.STALE_RESULT_SECONDS:
                if self._showing_hand:
                    self._clear_hand("No recent frames")  # Never leave stale tracking on screen.
                if self.controller.pointer is not None or self.controller.canvas.is_drawing:
                    self.controller.tracking_lost()
                    self._after_canvas_change()
        return len(snapshots)

    def shutdown(self) -> None:
        """Cancel callbacks, stop the worker, release the camera, close the window. Idempotent."""
        if self._closing:
            return
        self._closing = True
        if self._poll_job is not None:
            try:
                self.after_cancel(self._poll_job)
            except Exception:
                pass
            self._poll_job = None
        try:
            self.controller.reset_interaction()
        finally:
            try:
                self._source.stop()
            finally:
                try:
                    self.destroy()
                except Exception:
                    pass

    # --- rendering -------------------------------------------------------------
    def render(self, snap: FrameSnapshot) -> None:
        self._set(self._camera_dot, text_color=_CAMERA_DOT.get(snap.camera_state, C["warning"]))
        self._set(self._camera_label, text=_camera_pill_text(snap))
        self._set(self._detail_label, text=snap.detail)
        self._set(self._fps_value, text=f"{snap.fps:.1f} FPS" if snap.fps is not None else "— FPS")

        if snap.frame_rgb is not None:
            self._last_frame = snap.frame_rgb
        elif snap.camera_state is not CameraState.CONNECTED:
            self._last_frame = None  # Don't leave a frozen frame up when the camera is gone.
            self._preview.itemconfigure(self._placeholder, text=snap.camera_message)
        self._redraw_last_frame()

        if self.mode is Mode.AIR_CANVAS:
            self._update_canvas_panel()
        else:
            self._render_hand_panel(snap)

    def _update_canvas_panel(self) -> None:
        snap = self._last_snapshot
        if snap is None:
            text, color, count = "Waiting for camera", C["warning"], None
        else:
            text, color = snap.tracking_state.value, _TRACKING_COLOR[snap.tracking_state]
            count = snap.fingers.extended_count if snap.fingers is not None else None
        self._canvas_panel.update_from(self.controller, text, color, count, time.monotonic())

    def _set(self, widget, **kwargs) -> None:
        """configure() only when a value changed: CTk redraws a widget on every configure."""
        changed = {k: v for k, v in kwargs.items() if self._configured.get((id(widget), k)) != v}
        if changed:
            widget.configure(**changed)
            for k, v in changed.items():
                self._configured[(id(widget), k)] = v

    def _render_hand_panel(self, snap: FrameSnapshot) -> None:
        color = _TRACKING_COLOR[snap.tracking_state]
        self._set(self._tracking_label, text=snap.tracking_state.value, text_color=color)
        self._set(self._tracking_dot, text_color=color)
        if snap.hand is None or snap.fingers is None:
            self._clear_hand()
            return
        self._showing_hand = True
        hand = snap.hand
        if hand.handedness:
            score = f" · {hand.handedness_score:.0%}" if hand.handedness_score is not None else ""
            self._set(self._hand_value, text=f"{hand.handedness}{score}")
        else:
            self._set(self._hand_value, text="Unknown")
        self._set(self._landmarks_value, text=str(len(hand.landmarks)))

        fingers = snap.fingers
        count = fingers.extended_count
        self._set(self._count_label, 
            text="—" if count is None else str(count),
            text_color=C["text_muted"] if count is None else C["text"],
        )
        unsure = fingers.uncertain_count
        self._set(self._uncertain_label, text=f"{unsure} uncertain" if unsure else "")
        for finger, chip in self._finger_chips.items():
            text, fg, fg_text = _FINGER_STYLE[fingers.state_of(finger)]
            self._set(chip, text=text, fg_color=fg, text_color=fg_text)

    def _clear_hand(self, reason: Optional[str] = None) -> None:
        self._showing_hand = False
        self._set(self._count_label, text="—", text_color=C["text_muted"])
        self._set(self._uncertain_label, text="")
        self._set(self._hand_value, text="—")
        self._set(self._landmarks_value, text="—")
        for chip in self._finger_chips.values():
            self._set(chip, text="—", fg_color=C["elevated"], text_color=C["text_muted"])
        if reason:
            self._set(self._tracking_label, text=reason, text_color=C["warning"])
            self._set(self._tracking_dot, text_color=C["warning"])

    def _on_preview_resize(self, event: tk.Event) -> None:
        self._preview.coords(self._placeholder, event.width / 2, event.height / 2)
        self._redraw_last_frame()

    def _redraw_last_frame(self) -> None:
        if self._closing:
            return
        frame = self._last_frame
        if frame is None and self.mode is Mode.AIR_CANVAS:
            # Keep the canvas usable with mouse/keyboard even without a camera.
            frame = np.empty((*_BLANK_CANVAS, 3), np.uint8)
            frame[:] = config.hex_to_rgb(C["blank_canvas"])
        if frame is None:
            self._preview.itemconfigure(self._image_item, state="hidden")
            self._preview.itemconfigure(self._placeholder, state="normal")
            self._image_visible = False
            self._clear_overlays()
            self._viewport = None
            return
        self._show_frame(frame)

    def _show_frame(self, frame_rgb: np.ndarray) -> None:
        """Fit the frame into the preview (aspect kept, letterboxed) and add overlays."""
        area_w, area_h = self._preview.winfo_width(), self._preview.winfo_height()
        h, w = frame_rgb.shape[:2]
        viewport = fit_viewport(w, h, area_w, area_h)
        if viewport is None or area_w < 20 or area_h < 20:
            return  # Not laid out yet; <Configure> will redraw.
        self._viewport = viewport
        size = (viewport.display_w, viewport.display_h)
        interp = cv2.INTER_AREA if viewport.display_w < w else cv2.INTER_LINEAR
        image = cv2.resize(frame_rgb, size, interpolation=interp)
        if self.mode is Mode.AIR_CANVAS:
            image = composite(image, self._layer_cache.layer_for(self.controller.canvas, *size))
        pil_image = Image.fromarray(np.ascontiguousarray(image))
        if self._photo is not None and (self._photo.width(), self._photo.height()) == size:
            self._photo.paste(pil_image)  # Reuse the Tk image: much cheaper than recreating it.
        else:
            self._photo = ImageTk.PhotoImage(pil_image)
            self._preview.itemconfigure(self._image_item, image=self._photo)
        if not self._image_visible:
            self._preview.itemconfigure(self._image_item, state="normal")
            self._preview.itemconfigure(self._placeholder, state="hidden")
            self._image_visible = True
        self._preview.coords(self._image_item, viewport.offset_x, viewport.offset_y)
        self.frames_rendered += 1
        if self.mode is Mode.AIR_CANVAS:
            self._draw_toolbar(viewport)
            self._draw_cursor(viewport)
            if self.diagnostics:
                self._draw_diagnostics(viewport)

    # --- diagnostics (optional; toggled with D or the Debug switch) --------------------
    def set_diagnostics(self, enabled: bool) -> None:
        self.diagnostics = enabled
        if not enabled:
            self._preview.delete("diag")
            self._diag_items = None
        if self._canvas_panel.debug_switch.get() != int(enabled):
            self._canvas_panel.debug_switch.select() if enabled else self._canvas_panel.debug_switch.deselect()
        self._redraw_last_frame()

    def diagnostics_text(self) -> str:
        ctl, vp = self.controller, self._viewport
        fmt = lambda p: "—" if p is None else f"{p[0]:.3f}, {p[1]:.3f}"
        mapped = "—"
        if ctl.pointer is not None and vp is not None:
            x, y = vp.to_display(*ctl.pointer)
            mapped = f"{x:.0f}, {y:.0f} px"
        hovered = ctl.hovered.label if ctl.hovered else "—"
        reason = ctl.raw_reason or "ok"
        zone = ctl.zone.value if ctl.zone else "—"
        lines = [
            f"raw: {ctl.raw_gesture.value} ({reason})   stable: {ctl.gesture.value}",
            f"zone: {zone}   mode: {ctl.interaction}   speed: {ctl.pointer_speed:.2f}/s",
            f"tip raw: {fmt(ctl.raw_pointer)}   filtered: {fmt(ctl.pointer)}",
            f"mapped: {mapped}" + (f"   image {vp.display_w}x{vp.display_h} @ {vp.offset_x},{vp.offset_y}" if vp else ""),
            f"hover: {hovered}   dwell: {ctl.dwell_progress:.0%}",
            f"status: {ctl.status or '—'}",
        ]
        summary = ctl.recent_summary()
        if summary:
            lines.append("last 3 s: " + summary)
        return "\n".join(lines)

    def _draw_diagnostics(self, vp: Viewport) -> None:
        cv = self._preview
        if self._diag_items is None:
            box = cv.create_rectangle(0, 0, 0, 0, fill=C["background"], outline=C["border_strong"],
                                      tags=("overlay", "diag"))
            text = cv.create_text(0, 0, anchor="sw", fill=C["text_secondary"], font=(theme.mono_family(self), 9),
                                  tags=("overlay", "diag"))
            self._diag_items = (box, text)
        box, text = self._diag_items
        cv.itemconfigure(text, text=self.diagnostics_text())
        cv.coords(text, vp.offset_x + 10, vp.offset_y + vp.display_h - 8)
        x0, y0, x1, y1 = cv.bbox(text)
        cv.coords(box, x0 - 6, y0 - 4, x1 + 6, y1 + 4)
        cv.tag_raise(box)
        cv.tag_raise(text)

    # --- Air Canvas overlays (drawn as tk canvas items above the image) -----------------
    def _rounded_rect(self, x0: float, y0: float, x1: float, y1: float, r: float,
                      tags: tuple[str, ...] = ("overlay", "toolbar"), **kw) -> int:
        pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
               x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
        return self._preview.create_polygon(pts, smooth=True, tags=tags, **kw)

    def _clear_overlays(self) -> None:
        self._preview.delete("overlay")
        self._diag_items = None
        self._toolbar_signature = None
        self._cursor_items = None

    def _draw_toolbar(self, vp: Viewport) -> None:
        """Redraw the virtual toolbar only when something visible changed."""
        ctl = self.controller
        now = time.monotonic()
        signature = (
            vp, ctl.hovered.label if ctl.hovered else None, round(ctl.dwell_progress, 2),
            ctl.color_name, ctl.size_name, ctl.canvas.tool, ctl.clear_armed(now),
        )
        if signature == self._toolbar_signature:
            return
        self._toolbar_signature = signature
        self._preview.delete("toolbar")
        items = ctl.toolbar
        pad = 0.008
        bx0, by0 = vp.to_display(items[0].rect[0] - pad, items[0].rect[1] - pad)
        bx1, by1 = vp.to_display(items[-1].rect[2] + pad, items[-1].rect[3] + pad)
        self._rounded_rect(bx0, by0, bx1, by1, theme.RADIUS_LG, fill=C["surface"], outline=C["border_strong"])
        eraser_on = ctl.canvas.tool is Tool.ERASER
        # Label size follows the UI type scale (DPI aware), never the toolbar height.
        # CTk fonts are pixel sized x DPI scaling; use the caption size, capped by item height.
        scaling = ctk.ScalingTracker.get_widget_scaling(self)
        item_h = (items[0].rect[3] - items[0].rect[1]) * vp.display_h
        font_px = max(9, min(round(theme.CAPTION * scaling), int(item_h * 0.3)))
        for index, item in enumerate(items):
            x0, y0 = vp.to_display(item.rect[0], item.rect[1])
            x1, y1 = vp.to_display(item.rect[2], item.rect[3])
            self._draw_toolbar_item(item, x0, y0, x1, y1, eraser_on, now, index, font_px)

    def _draw_toolbar_item(self, item: ToolbarItem, x0: float, y0: float, x1: float, y1: float,
                           eraser_on: bool, now: float, index: int, font_px: int) -> None:
        ctl, cv = self.controller, self._preview
        action = item.command.action
        hovered = ctl.hovered is item
        active = (action is Action.COLOR and item.command.value == ctl.color_name and not eraser_on) or (
            action is Action.ERASER and eraser_on)
        armed = action is Action.CLEAR and ctl.clear_armed(now)
        if armed:
            fill, outline, width = C["elevated"], C["warning"], 2
        elif active:
            fill, outline, width = (C["accent_muted"] if action is Action.ERASER else C["elevated"]), C["accent"], 2
        elif hovered:
            fill, outline, width = C["elevated"], C["border_strong"], 1
        else:
            fill, outline, width = C["surface"], C["surface"], 1
        self._rounded_rect(x0, y0, x1, y1, theme.RADIUS_MD, tags=("overlay", "toolbar", f"tb{index}"),
                           fill=fill, outline=outline, width=width)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if action is Action.COLOR:
            r = min(x1 - x0, y1 - y0) * 0.24
            cv.create_oval(cx - r, cy - r, cx + r, cy + r, fill=config.BRUSH_COLORS[item.command.value],
                           outline=C["background"], tags=("overlay", "toolbar"))
        else:
            label = {Action.CYCLE_SIZE: ctl.size_name, Action.CLEAR: "Confirm" if armed else "Clear"}.get(
                action, item.label)
            if armed:
                color = C["warning"]
            elif active:
                color = C["accent"]
            else:
                color = C["text"] if hovered else C["text_secondary"]
            cv.create_text(cx, cy, text=label, fill=color, font=(theme.semibold_family(self), -font_px),
                           tags=("overlay", "toolbar"))
        if hovered and ctl.dwell_progress > 0:
            # Dwell progress: a thin bar along the bottom edge of the item.
            bar = C["warning"] if armed else C["accent"]
            px1 = x0 + 6 + (x1 - x0 - 12) * ctl.dwell_progress
            cv.create_line(x0 + 6, y1 - 4, px1, y1 - 4, fill=bar, width=3, capstyle="round",
                           tags=("overlay", "toolbar"))

    def _draw_cursor(self, vp: Viewport) -> None:
        ctl, cv = self.controller, self._preview
        if self._cursor_items is None:
            ring = cv.create_oval(0, 0, 0, 0, width=2, tags=("overlay", "cursor"))
            dot = cv.create_oval(0, 0, 0, 0, outline="", tags=("overlay", "cursor"))
            # Dwell progress, drawn right around the fingertip where the user is looking.
            arc = cv.create_arc(0, 0, 0, 0, start=90, extent=0, style="arc", width=3,
                                outline=C["accent"], tags=("overlay", "cursor"))
            self._cursor_items = (ring, dot, arc)
        ring, dot, arc = self._cursor_items
        if ctl.pointer is None:
            cv.itemconfigure("cursor", state="hidden")
            return
        x, y = vp.to_display(*ctl.pointer)
        gesture = ctl.gesture
        eraser_on = ctl.canvas.tool is Tool.ERASER
        radius = max(6.0, ctl.canvas.effective_thickness * vp.display_h / 2 + 3)
        selecting = ctl.zone is Zone.TOOLBAR and gesture in (Gesture.DRAW, Gesture.SELECT)
        if selecting:
            color, radius = C["accent"], 10
        elif gesture is Gesture.DRAW and ctl.zone is Zone.CANVAS:
            color = C["text"] if eraser_on else config.BRUSH_COLORS[ctl.color_name]
        else:
            color = C["text_muted"]
        drawing_look = gesture is Gesture.DRAW and ctl.zone is Zone.CANVAS
        dash = "" if selecting or (drawing_look and not eraser_on) else (4, 3)
        cv.coords(ring, x - radius, y - radius, x + radius, y + radius)
        cv.itemconfigure(ring, outline=color, dash=dash, state="normal")
        cv.coords(dot, x - 2, y - 2, x + 2, y + 2)
        cv.itemconfigure(dot, fill=color, state="normal")
        if selecting and ctl.hovered is not None and ctl.dwell_progress > 0:
            r = radius + 5
            cv.coords(arc, x - r, y - r, x + r, y + r)
            extent = -359.9 * ctl.dwell_progress  # Clockwise from 12 o'clock.
            confirming = ctl.hovered.command.action is Action.CLEAR and ctl.clear_armed(time.monotonic())
            cv.itemconfigure(arc, extent=extent, state="normal",
                             outline=C["warning"] if confirming else C["accent"])
        else:
            cv.itemconfigure(arc, state="hidden")
        cv.tag_raise("cursor")
