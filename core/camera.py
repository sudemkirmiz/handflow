"""Webcam access through OpenCV: open, read, release.

OpenCV cannot reliably tell *why* a camera failed to open (missing device,
used by another application, or blocked by OS privacy settings), so all of
those are reported as one honest ``UNAVAILABLE`` state with a helpful message.
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import Any, Callable, Optional

import cv2
import numpy as np

CaptureFactory = Callable[[int, int], Any]


class CameraState(Enum):
    DISCONNECTED = "disconnected"  # Not opened yet, or released.
    CONNECTED = "connected"
    UNAVAILABLE = "unavailable"  # Could not open the device.
    READ_FAILED = "read_failed"  # Opened, but frames stopped arriving.


UNAVAILABLE_MESSAGE = (
    "Camera unavailable — not connected, in use by another app, "
    "or blocked in Windows privacy settings."
)


def _default_capture_factory(index: int, backend: int) -> Any:
    return cv2.VideoCapture(index, backend)


def _default_backends() -> tuple[int, ...]:
    # DirectShow opens much faster than Media Foundation on most Windows webcams.
    if sys.platform.startswith("win"):
        return (cv2.CAP_DSHOW, cv2.CAP_ANY)
    return (cv2.CAP_ANY,)


class Camera:
    """Thin, testable wrapper around ``cv2.VideoCapture``."""

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        max_consecutive_failures: int = 15,
        capture_factory: CaptureFactory = _default_capture_factory,
        backends: Optional[tuple[int, ...]] = None,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.max_consecutive_failures = max_consecutive_failures
        self._capture_factory = capture_factory
        self._backends = backends if backends is not None else _default_backends()
        self._capture: Any = None
        self._failures = 0
        self.state = CameraState.DISCONNECTED
        self.message = "Camera not started"

    @property
    def is_open(self) -> bool:
        return self._capture is not None

    def open(self) -> bool:
        """Try each backend until one delivers a real frame."""
        self.release()
        for backend in self._backends:
            capture = None
            try:
                capture = self._capture_factory(self.index, backend)
                if capture is None or not capture.isOpened():
                    raise RuntimeError("not opened")
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                ok, frame = capture.read()  # A device can "open" but deliver nothing.
                if not ok or frame is None:
                    raise RuntimeError("no first frame")
            except Exception:
                if capture is not None:
                    capture.release()
                continue
            self._capture = capture
            self._failures = 0
            h, w = frame.shape[:2]
            self.state = CameraState.CONNECTED
            self.message = f"Connected · {w}×{h}"
            return True

        self.state = CameraState.UNAVAILABLE
        self.message = UNAVAILABLE_MESSAGE
        return False

    def read(self) -> Optional[np.ndarray]:
        """Return a BGR frame, or ``None`` if this read failed."""
        if self._capture is None:
            return None
        try:
            ok, frame = self._capture.read()
        except Exception:
            ok, frame = False, None
        if ok and frame is not None:
            self._failures = 0
            if self.state is not CameraState.CONNECTED:
                h, w = frame.shape[:2]
                self.state = CameraState.CONNECTED
                self.message = f"Connected · {w}×{h}"
            return frame

        self._failures += 1
        if self._failures >= self.max_consecutive_failures:
            self.release()
            self.state = CameraState.READ_FAILED
            self.message = "Camera stopped sending frames — reconnecting…"
        return None

    def release(self) -> None:
        """Release the device. Safe to call any number of times."""
        capture, self._capture = self._capture, None
        if capture is not None:
            try:
                capture.release()
            finally:
                self.state = CameraState.DISCONNECTED
                self.message = "Camera released"

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
