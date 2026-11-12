import numpy as np

from core.camera import Camera, CameraState


class FakeCapture:
    def __init__(self, opened=True, frames=None):
        self.opened = opened
        self.frames = list(frames) if frames is not None else None
        self.release_calls = 0
        self.props = {}

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def read(self):
        if self.frames is None:
            return True, np.zeros((480, 640, 3), np.uint8)
        if self.frames:
            frame = self.frames.pop(0)
            return (frame is not None), frame
        return False, None

    def release(self):
        self.release_calls += 1


class Factory:
    def __init__(self, captures):
        self.captures = list(captures)
        self.created = []

    def __call__(self, index, backend):
        cap = self.captures.pop(0)
        self.created.append(cap)
        return cap


def test_no_camera_available_reports_unavailable_and_releases_attempts():
    factory = Factory([FakeCapture(opened=False), FakeCapture(opened=False)])
    cam = Camera(capture_factory=factory, backends=(1, 2))
    assert cam.open() is False
    assert cam.state is CameraState.UNAVAILABLE
    assert "unavailable" in cam.message.lower()
    assert not cam.is_open
    assert cam.read() is None
    assert all(c.release_calls == 1 for c in factory.created)


def test_camera_that_opens_but_gives_no_frames_is_unavailable():
    factory = Factory([FakeCapture(frames=[])])
    cam = Camera(capture_factory=factory, backends=(1,))
    assert cam.open() is False
    assert factory.created[0].release_calls == 1


def test_factory_exception_is_handled():
    def boom(index, backend):
        raise RuntimeError("driver crashed")

    cam = Camera(capture_factory=boom, backends=(1,))
    assert cam.open() is False
    assert cam.state is CameraState.UNAVAILABLE


def test_falls_back_to_next_backend():
    good = FakeCapture()
    factory = Factory([FakeCapture(opened=False), good])
    cam = Camera(capture_factory=factory, backends=(1, 2))
    assert cam.open() is True
    assert cam.state is CameraState.CONNECTED
    assert "640×480" in cam.message
    assert cam.read().shape == (480, 640, 3)


def test_release_is_idempotent_and_releases_once():
    cap = FakeCapture()
    cam = Camera(capture_factory=Factory([cap]), backends=(1,))
    cam.open()
    cam.release()
    cam.release()
    assert cap.release_calls == 1
    assert not cam.is_open
    assert cam.state is CameraState.DISCONNECTED


def test_context_manager_releases_camera():
    cap = FakeCapture()
    with Camera(capture_factory=Factory([cap]), backends=(1,)) as cam:
        assert cam.open()
    assert cap.release_calls == 1


def test_repeated_read_failures_release_camera():
    frame = np.zeros((480, 640, 3), np.uint8)
    cap = FakeCapture(frames=[frame] + [None] * 5)
    cam = Camera(capture_factory=Factory([cap]), backends=(1,), max_consecutive_failures=3)
    assert cam.open()
    assert cam.read() is None and cam.is_open  # A single failed read is tolerated.
    assert cam.read() is None and cam.is_open
    assert cam.read() is None
    assert not cam.is_open
    assert cam.state is CameraState.READ_FAILED
    assert cap.release_calls == 1


def test_reopen_releases_previous_capture():
    first, second = FakeCapture(), FakeCapture()
    cam = Camera(capture_factory=Factory([first, second]), backends=(1,))
    cam.open()
    cam.open()
    assert first.release_calls == 1
    assert second.release_calls == 0
    cam.release()
    assert second.release_calls == 1
