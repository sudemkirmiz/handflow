"""Air Canvas inspector: live hand status, brush state, export.

The virtual toolbar on the camera image is the primary control surface (hand
gestures *and* mouse clicks). The inspector therefore does not repeat its
buttons; it shows state, offers a direct brush-size choice (the toolbar only
cycles sizes) and holds the export controls. Every control calls the same
``AirCanvasController.execute`` path used by gestures.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import customtkinter as ctk

import config
from core.air_canvas import Action, AirCanvasController, Command
from core.canvas import Tool
from ui import theme
from ui.theme import C

_MODE_COLOR = {
    "Drawing": C["text"],
    "Selecting": C["accent"],
    "Pointing": C["text"],
    "Unclear": C["warning"],
}

SHORTCUTS = "1–6 color · Q W E size · X eraser\nCtrl+Z undo · Ctrl+S save · Del clear · D diagnostics"


class AirCanvasPanel(ctk.CTkFrame):
    def __init__(
        self,
        parent: Any,
        fonts: theme.Fonts,
        on_command: Callable[[Command], None],
        on_debug: Callable[[bool], None] = lambda _enabled: None,
    ) -> None:
        super().__init__(parent, fg_color=C["surface"], corner_radius=theme.RADIUS_LG, border_width=1,
                         border_color=C["border"])
        self._fonts = fonts
        self._on_command = on_command
        self._cache: dict[tuple[int, str], Any] = {}
        self.background_name = "Transparent"
        pad = {"padx": theme.SPACE_LG}

        # --- Hand -------------------------------------------------------------------
        theme.section_label(self, "Hand", fonts).pack(fill="x", pady=(theme.SPACE_LG, theme.SPACE_XS), **pad)
        status_row = self._row("Status")
        self.tracking_label = ctk.CTkLabel(status_row, text="Waiting for camera", font=fonts.body_strong,
                                           text_color=C["warning"], height=24)
        self.tracking_label.pack(side="right")
        self.tracking_dot = ctk.CTkLabel(status_row, text="●", font=fonts.caption, text_color=C["warning"],
                                         width=12, height=24)
        self.tracking_dot.pack(side="right", padx=(0, theme.SPACE_XS))
        self.mode_value = self._value(self._row("Mode"))
        self.fingers_value = self._value(self._row("Fingers up"))

        theme.divider(self).pack(fill="x", pady=theme.SPACE_MD, **pad)

        # --- Brush ------------------------------------------------------------------
        theme.section_label(self, "Brush", fonts).pack(fill="x", pady=(0, theme.SPACE_XS), **pad)
        tool_row = self._row("Tool")
        self.tool_value = self._value(tool_row)
        self.tool_swatch = ctk.CTkLabel(tool_row, text="●", font=fonts.title, text_color=C["text"], width=14,
                                        height=24)
        self.tool_swatch.pack(side="right", padx=(0, theme.SPACE_XS))
        ctk.CTkLabel(self, text="Size", font=fonts.body, text_color=C["text_secondary"], anchor="w",
                     height=20).pack(fill="x", pady=(theme.SPACE_XS, theme.SPACE_XS), **pad)
        self.size_buttons = theme.segmented(self, list(config.BRUSH_SIZES), fonts,
                                            lambda v: on_command(Command(Action.SIZE, v)))
        self.size_buttons.pack(fill="x", **pad)

        theme.divider(self).pack(fill="x", pady=theme.SPACE_MD, **pad)

        # --- Export -----------------------------------------------------------------
        theme.section_label(self, "Export", fonts).pack(fill="x", pady=(0, theme.SPACE_XS), **pad)
        background_row = ctk.CTkFrame(self, fg_color="transparent", height=20)
        background_row.pack(fill="x", pady=(0, theme.SPACE_XS), **pad)
        ctk.CTkLabel(background_row, text="Export Background", font=fonts.body, text_color=C["text_secondary"],
                     height=20).pack(side="left")
        ctk.CTkLabel(background_row, text="PNG only", font=fonts.caption, text_color=C["text_muted"],
                     height=20).pack(side="right")
        self.background_buttons = theme.segmented(self, list(config.EXPORT_BACKGROUNDS), fonts,
                                                  self._set_background)
        self.background_buttons.set(self.background_name)
        self.background_buttons.pack(fill="x", **pad)
        self.save_button = theme.primary_button(self, "Save PNG", fonts,
                                                lambda: on_command(Command(Action.SAVE)))
        self.save_button.configure(text_color_disabled=C["text_muted"])
        self.save_button.pack(fill="x", pady=(theme.SPACE_SM, 0), **pad)

        self.message_label = ctk.CTkLabel(self, text="", font=fonts.caption, text_color=C["text_secondary"],
                                          wraplength=theme.INSPECTOR_WIDTH - 2 * theme.SPACE_LG,
                                          justify="left", anchor="w")
        self.message_label.pack(fill="x", pady=(theme.SPACE_SM, 0), **pad)

        # --- Footer: shortcuts + diagnostics (advanced, off by default) ---------------
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(side="bottom", fill="x", pady=(0, theme.SPACE_MD), **pad)
        theme.divider(footer).pack(fill="x", pady=(0, theme.SPACE_SM))
        ctk.CTkLabel(footer, text=SHORTCUTS, font=fonts.caption, text_color=C["text_muted"], justify="left",
                     anchor="w").pack(fill="x")
        self.debug_switch = ctk.CTkSwitch(
            footer, text="Diagnostics overlay", font=fonts.caption, text_color=C["text_secondary"],
            switch_width=28, switch_height=14, progress_color=C["accent"], button_color=C["text"],
            button_hover_color=C["text"], fg_color=C["elevated"],
            command=lambda: on_debug(bool(self.debug_switch.get())),
        )
        self.debug_switch.pack(anchor="w", pady=(theme.SPACE_SM, 0))

    # --- construction helpers ------------------------------------------------------
    def _row(self, name: str) -> ctk.CTkFrame:
        row = ctk.CTkFrame(self, fg_color="transparent", height=24)
        row.pack(fill="x", padx=theme.SPACE_LG)
        ctk.CTkLabel(row, text=name, font=self._fonts.body, text_color=C["text_secondary"], height=24).pack(
            side="left")
        return row

    def _value(self, row: ctk.CTkFrame) -> ctk.CTkLabel:
        value = ctk.CTkLabel(row, text="—", font=self._fonts.body_strong, text_color=C["text"], height=24)
        value.pack(side="right")
        return value

    def _set_background(self, name: str) -> None:
        self.background_name = name

    def _set(self, widget: Any, **kwargs: Any) -> None:
        """configure() only when a value really changed (CTk redraws are not free)."""
        changed = {k: v for k, v in kwargs.items() if self._cache.get((id(widget), k)) != v}
        if changed:
            widget.configure(**changed)
            for k, v in changed.items():
                self._cache[(id(widget), k)] = v

    # --- live updates --------------------------------------------------------------
    def update_from(self, controller: AirCanvasController, tracking_text: str, tracking_color: str,
                    finger_count: Optional[int], now: float) -> None:
        mode = controller.interaction
        self._set(self.tracking_label, text=tracking_text, text_color=tracking_color)
        self._set(self.tracking_dot, text_color=tracking_color)
        self._set(self.mode_value, text=mode, text_color=_MODE_COLOR.get(mode, C["text_secondary"]))
        self._set(self.fingers_value, text="—" if finger_count is None else str(finger_count))

        eraser = controller.canvas.tool is Tool.ERASER
        self._set(self.tool_value, text="Eraser" if eraser else controller.color_name)
        self._set(self.tool_swatch, text="◌" if eraser else "●",
                  text_color=C["text"] if eraser else config.BRUSH_COLORS[controller.color_name])
        if self.size_buttons.get() != controller.size_name:
            self.size_buttons.set(controller.size_name)
        self._set(self.save_button, state="disabled" if controller.canvas.is_empty else "normal")
        self._set(self.message_label, text=controller.message)
