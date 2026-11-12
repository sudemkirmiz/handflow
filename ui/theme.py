"""Design tokens for HandFlow Studio: spacing, radii, control sizes, type scale, fonts.

Colors live in ``config.COLORS`` (single source); everything about *how* the UI
is laid out and typeset lives here, so the look stays consistent.
"""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from typing import Optional

import customtkinter as ctk

import config

C = config.COLORS

# 8 px based spacing scale (4 for tight inner gaps).
SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 12
SPACE_LG = 16
SPACE_XL = 24

RADIUS_SM = 4  # Chips, segmented items.
RADIUS_MD = 6  # Buttons, inputs, toolbar items.
RADIUS_LG = 8  # Panels.

CONTROL_SM = 28
CONTROL_MD = 32

HEADER_HEIGHT = 52
INSPECTOR_WIDTH = 288

# Type scale (points; CustomTkinter applies the Windows DPI scaling).
CAPTION = 11
BODY = 12
TITLE = 13
BRAND = 15
METRIC = 28

_UI_FAMILIES = ("Segoe UI Variable Text", "Segoe UI", "Helvetica Neue", "Arial")
_DISPLAY_FAMILIES = ("Segoe UI Variable Display", "Segoe UI", "Helvetica Neue", "Arial")
_SEMIBOLD_FAMILIES = ("Segoe UI Semibold", "Segoe UI Variable Text", "Segoe UI", "Arial")
_MONO_FAMILIES = ("Cascadia Mono", "Consolas", "Menlo", "Courier New")
_cache: dict[str, str] = {}


def _pick(kind: str, candidates: tuple[str, ...], root: Optional[tk.Misc]) -> str:
    if kind not in _cache:
        try:
            available = set(tkfont.families(root))
        except tk.TclError:
            available = set()
        _cache[kind] = next((f for f in candidates if f in available), candidates[-1])
    return _cache[kind]


def ui_family(root: Optional[tk.Misc] = None) -> str:
    return _pick("ui", _UI_FAMILIES, root)


def display_family(root: Optional[tk.Misc] = None) -> str:
    return _pick("display", _DISPLAY_FAMILIES, root)


def semibold_family(root: Optional[tk.Misc] = None) -> str:
    return _pick("semibold", _SEMIBOLD_FAMILIES, root)


def mono_family(root: Optional[tk.Misc] = None) -> str:
    return _pick("mono", _MONO_FAMILIES, root)


class Fonts:
    """CTkFont objects are created once and shared (no per-frame font creation)."""

    def __init__(self, root: tk.Misc) -> None:
        ui, display, semi = ui_family(root), display_family(root), semibold_family(root)
        self.caption = ctk.CTkFont(ui, CAPTION)
        self.caption_strong = ctk.CTkFont(semi, CAPTION)
        self.body = ctk.CTkFont(ui, BODY)
        self.body_strong = ctk.CTkFont(semi, BODY)
        self.title = ctk.CTkFont(semi, TITLE)
        self.brand = ctk.CTkFont(display, BRAND, "bold")
        self.metric = ctk.CTkFont(display, METRIC, "bold")


def divider(parent: ctk.CTkBaseClass) -> ctk.CTkFrame:
    return ctk.CTkFrame(parent, height=1, corner_radius=0, fg_color=C["border"])


def panel(parent: ctk.CTkBaseClass, **kwargs) -> ctk.CTkFrame:
    return ctk.CTkFrame(parent, fg_color=C["surface"], corner_radius=RADIUS_LG, border_width=1,
                        border_color=C["border"], **kwargs)


def segmented(parent: ctk.CTkBaseClass, values: list[str], fonts: Fonts, command) -> ctk.CTkSegmentedButton:
    return ctk.CTkSegmentedButton(
        parent, values=values, command=command, height=CONTROL_SM, corner_radius=RADIUS_MD,
        font=fonts.body, fg_color=C["elevated"], text_color=C["text"],
        selected_color=C["accent_muted"], selected_hover_color=C["accent_muted"],
        unselected_color=C["elevated"], unselected_hover_color=C["border_strong"],
    )


def primary_button(parent: ctk.CTkBaseClass, text: str, fonts: Fonts, command) -> ctk.CTkButton:
    return ctk.CTkButton(
        parent, text=text, command=command, height=CONTROL_MD, corner_radius=RADIUS_MD, font=fonts.body_strong,
        fg_color=C["accent"], hover_color=C["accent_hover"], text_color=C["on_accent"],
    )


def section_label(parent: ctk.CTkBaseClass, text: str, fonts: Fonts) -> ctk.CTkLabel:
    return ctk.CTkLabel(parent, text=text.upper(), font=fonts.caption_strong, text_color=C["text_muted"],
                        anchor="w", height=18)
