"""Camera discovery and frame capture.

Colour-order note (the classic Picamera2 trap): libcamera names formats in
little-endian byte order, so requesting ``"RGB888"`` hands back a numpy array
whose channels are ordered **B, G, R**. That happens to be exactly what OpenCV
and Ultralytics expect, so every frame in this project is BGR end to end --
capture, JPEG encode, and the client's ``imshow``. Do not "fix" this by
switching to ``"BGR888"``; that would give RGB and invert every preview.

Exposure note: with no controls set, the AE loop is free to choose a long
shutter, and indoors it does. A capture session of a moving dog produced crops
whose sharpness (variance of Laplacian) measured 2-35 against 36 for the static
background -- i.e. the subject was smeared to the point of being useless for
identification, while the lens was perfectly in focus. `--exposure-mode short`
biases AE toward a short shutter and higher gain, which is the trade that
matters here: sensor noise costs an embedder far less than motion blur.
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


# Frames to pull before trusting the AE readout; convergence takes a beat.
AE_SETTLE_READS = 30

EXPOSURE_MODES = ("normal", "short", "long")
METERING_MODES = ("centre", "spot", "matrix")


def build_controls(
    exposure_mode: str = "normal",
    ev: float = 0.0,
    metering: str = "centre",
    exposure_us: int | None = None,
    gain: float | None = None,
) -> dict:
    """Translate our knobs into libcamera controls.

    Kept a pure function, and separate from `start()`, so the mapping can be
    tested without a camera -- the same reason `SyntheticSource` exists.

    `exposure_us` is the escape hatch, not the default: pinning the shutter
    means disabling AE entirely, and a fixed exposure that suits a sunlit room
    blacks out the same room at dusk. `exposure_mode="short"` keeps AE in charge
    and merely tells it which way to lean, which survives changing light.
    """
    from libcamera import controls as c

    if exposure_mode not in EXPOSURE_MODES:
        raise ValueError(f"exposure_mode must be one of {EXPOSURE_MODES}")
    if metering not in METERING_MODES:
        raise ValueError(f"metering must be one of {METERING_MODES}")

    if exposure_us is not None:
        # Full manual. Gain must be given too, or AE is off with nothing driving
        # the gain and the image is whatever the sensor defaults to.
        out = {"AeEnable": False, "ExposureTime": int(exposure_us)}
        if gain is not None:
            out["AnalogueGain"] = float(gain)
        return out

    out = {
        "AeEnable": True,
        "AeExposureMode": {
            "normal": c.AeExposureModeEnum.Normal,
            "short": c.AeExposureModeEnum.Short,
            "long": c.AeExposureModeEnum.Long,
        }[exposure_mode],
        "AeMeteringMode": {
            "centre": c.AeMeteringModeEnum.CentreWeighted,
            "spot": c.AeMeteringModeEnum.Spot,
            "matrix": c.AeMeteringModeEnum.Matrix,
        }[metering],
    }
    if ev:
        # Positive EV lifts a subject that AE has underexposed to protect a
        # bright background -- a backlit doorway metered the dog to 50/255 here.
        out["ExposureValue"] = float(ev)
    if gain is not None:
        out["AnalogueGain"] = float(gain)
    return out


class PiCameraSource:
    """A single Picamera2 device configured for continuous video capture."""

    def __init__(
        self,
        info: CameraInfo,
        size: tuple[int, int],
        buffer_count: int = 4,
        controls: dict | None = None,
    ):
        self.info = info
        self.size = size
        self._buffer_count = buffer_count
        self._controls = controls or {}
        self._picam = None

    def start(self) -> None:
        from picamera2 import Picamera2

        picam = Picamera2(camera_num=self.info.num)
        config = picam.create_video_configuration(
            main={"size": self.size, "format": "RGB888"},  # -> BGR ndarray, see module docstring
            buffer_count=self._buffer_count,
            # Set at configure time rather than after start(), so the very first
            # frames already obey them instead of drifting in over the AE settle.
            controls=self._controls,
        )
        picam.configure(config)
        picam.start()
        # The AE/AWB loops need a moment; frames captured immediately are dark.
        time.sleep(0.5)
        self._picam = picam
        log.info("%s started at %dx%d", self.info.label, *self.size)
        if self._controls:
            log.info("%s controls: %s", self.info.label, self._controls)
        # What AE actually settled on -- the number worth having in the log when
        # the crops come back blurry. Read it repeatedly rather than once: the
        # AE loop needs a second or two to converge, and a single read straight
        # after start() reports a pre-settle value. It logged "33.0 ms" that way
        # while the stream was in fact producing sharp frames.
        try:
            for _ in range(AE_SETTLE_READS):
                md = picam.capture_metadata()
            log.info(
                "%s AE settled: exposure %.1f ms, analogue gain %.2f, lux %.0f",
                self.info.label,
                md.get("ExposureTime", 0) / 1000,
                md.get("AnalogueGain", 0),
                md.get("Lux", 0),
            )
        except Exception:  # metadata is diagnostics, never a reason to fail start
            log.debug("could not read camera metadata", exc_info=True)

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
