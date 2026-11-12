"""Capture a Tk window that is rendered off-screen, without ever showing or
activating it (used to make README screenshots from synthetic data).

The window is placed outside every monitor, is never focused, and its content
is read with the Win32 ``PrintWindow`` API. Windows only.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

from PIL import Image

PW_RENDERFULLCONTENT = 0x00000002
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD),
    ]


def offscreen_geometry(width: int, height: int) -> str:
    """A position left of the whole virtual desktop (all monitors)."""
    virtual_left = user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
    return f"{width}x{height}+{virtual_left - width - 400}+0"


def capture(hwnd: int, width: int, height: int) -> Image.Image:
    hdc_window = user32.GetWindowDC(hwnd)
    hdc_mem = gdi32.CreateCompatibleDC(hdc_window)
    bitmap = gdi32.CreateCompatibleBitmap(hdc_window, width, height)
    gdi32.SelectObject(hdc_mem, bitmap)
    try:
        if not user32.PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT):
            raise RuntimeError("PrintWindow failed")
        header = _BitmapInfoHeader()
        header.biSize = ctypes.sizeof(header)
        header.biWidth, header.biHeight = width, -height  # Top-down rows.
        header.biPlanes, header.biBitCount = 1, 32
        buffer = ctypes.create_string_buffer(width * height * 4)
        gdi32.GetDIBits(hdc_mem, bitmap, 0, height, buffer, ctypes.byref(header), 0)
        return Image.frombuffer("RGBA", (width, height), buffer, "raw", "BGRA", 0, 1).convert("RGB")
    finally:
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(hwnd, hdc_window)


def toplevel_hwnd(widget) -> int:
    """The native top-level window handle of a Tk root."""
    return user32.GetParent(widget.winfo_id()) or widget.winfo_id()


def foreground_window() -> int:
    return user32.GetForegroundWindow()


def foreground_is_ours() -> bool:
    """True if the active window belongs to this process (i.e. we stole focus).

    Comparing the foreground before/after is not enough: the user may switch
    windows meanwhile. Ownership is what matters.
    """
    import os

    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(pid))
    return pid.value == os.getpid()


GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020  # Mouse clicks pass through to whatever is below.
WS_EX_NOACTIVATE = 0x08000000  # Never becomes the foreground window.
WS_EX_TOOLWINDOW = 0x00000080  # No taskbar button.


def make_ghost(widget) -> None:
    """Invisible (alpha 0), click-through, never-activated window.

    It is still composed by Windows, so it lays out and can be captured with
    ``capture``, but the user can't see it, click it or lose focus to it.
    """
    widget.overrideredirect(True)
    widget.attributes("-alpha", 0.0)
    widget.update_idletasks()
    hwnd = toplevel_hwnd(widget)
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(
        hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
    )
