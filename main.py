"""HandFlow entry point."""

from __future__ import annotations

import os

# Quieten MediaPipe/TFLite native logging before it is imported.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import customtkinter as ctk

import config
from core.camera import Camera
from core.hand_tracker import HandTracker
from core.processing import FrameProcessor
from ui.app import HandFlowApp


def build_processor() -> FrameProcessor:
    camera = Camera(
        index=config.CAMERA_INDEX,
        width=config.CAMERA_WIDTH,
        height=config.CAMERA_HEIGHT,
        max_consecutive_failures=config.MAX_CONSECUTIVE_READ_FAILURES,
    )

    def make_tracker() -> HandTracker:
        return HandTracker.from_model(
            config.MODEL_PATH,
            num_hands=config.MAX_HANDS,
            min_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            # Frames are mirrored before detection, matching MediaPipe's selfie assumption.
            input_is_mirrored=config.MIRROR_PREVIEW,
            expected_sha256=config.MODEL_SHA256,
        )

    return FrameProcessor(
        camera,
        make_tracker,
        mirror=config.MIRROR_PREVIEW,
        retry_seconds=config.CAMERA_RETRY_SECONDS,
    )


def main() -> None:
    ctk.set_appearance_mode("dark")
    app = HandFlowApp(build_processor())
    try:
        app.start()
        app.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()  # Idempotent: safe even if the window already closed it.


if __name__ == "__main__":
    main()
