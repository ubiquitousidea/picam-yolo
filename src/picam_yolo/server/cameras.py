"""Camera discovery and frame capture.

Colour-order note (the classic Picamera2 trap): libcamera names formats in
little-endian byte order, so requesting ``"RGB888"`` hands back a numpy array
whose channels are ordered **B, G, R**. That happens to be exactly what OpenCV
and Ultralytics expect, so every frame in this project is BGR end to end --
capture, JPEG encode, and the client's ``imshow``. Do not "fix" this by
switching to ``"BGR888"``; that would give RGB and invert every preview.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraInfo:
    num: int
    model: str
    camera_id: str

    @property
    def label(self) -> str:
        return f"cam{self.num} ({self.model})"


class FrameSource(Protocol):
    """Anything that can hand out BGR frames."""

    info: CameraInfo

    def start(self) -> None: ...
    def read(self) -> np.ndarray: ...
    def stop(self) -> None: ...


def discover_cameras() -> list[CameraInfo]:
    """Enumerate every camera libcamera can see, in port order.

    Returns an empty list rather than raising when picamera2 is unavailable, so
    the caller can fall back to synthetic sources on a dev machine.
    """
    try:
        from picamera2 import Picamera2
    except ImportError:
        log.warning("picamera2 not importable; no real cameras will be used")
        return []

    found = []
    for idx, entry in enumerate(Picamera2.global_camera_info()):
        found.append(
            CameraInfo(
                num=entry.get("Num", idx),
                model=entry.get("Model", "unknown"),
                camera_id=entry.get("Id", ""),
            )
        )
    return found


class PiCameraSource:
    """A single Picamera2 device configured for continuous video capture."""

    def __init__(self, info: CameraInfo, size: tuple[int, int], buffer_count: int = 4):
        self.info = info
        self.size = size
        self._buffer_count = buffer_count
        self._picam = None

    def start(self) -> None:
        from picamera2 import Picamera2

        picam = Picamera2(camera_num=self.info.num)
        config = picam.create_video_configuration(
            main={"size": self.size, "format": "RGB888"},  # -> BGR ndarray, see module docstring
            buffer_count=self._buffer_count,
        )
        picam.configure(config)
        picam.start()
        # The AE/AWB loops need a moment; frames captured immediately are dark.
        time.sleep(0.5)
        self._picam = picam
        log.info("%s started at %dx%d", self.info.label, *self.size)

    def read(self) -> np.ndarray:
        if self._picam is None:
            raise RuntimeError("read() before start()")
        return self._picam.capture_array("main")

    def stop(self) -> None:
        if self._picam is not None:
            self._picam.stop()
            self._picam.close()
            self._picam = None


class SyntheticSource:
    """Moving-bars test pattern, so the client and wire format can be exercised
    on a machine with no cameras attached (``--synthetic``)."""

    def __init__(self, info: CameraInfo, size: tuple[int, int], fps: float = 30.0):
        self.info = info
        self.size = size
        self._period = 1.0 / fps
        self._t0 = 0.0
        self._n = 0

    def start(self) -> None:
        self._t0 = time.monotonic()

    def read(self) -> np.ndarray:
        # Pace ourselves so the pipeline sees a realistic frame interval.
        target = self._t0 + self._n * self._period
        now = time.monotonic()
        if target > now:
            time.sleep(target - now)
        self._n += 1

        w, h = self.size
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = int((self._n * 7) % max(w - 120, 1))
        frame[:, :, 0] = 40  # dim blue field
        frame[h // 4 : 3 * h // 4, x : x + 120] = (0, 165, 255)  # orange block
        return frame

    def stop(self) -> None:
        pass
