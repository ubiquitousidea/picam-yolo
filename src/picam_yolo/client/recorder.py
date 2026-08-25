"""Recording the incoming stream to video files, one per camera."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class StreamRecorder:
    """Writes frames to one video file per camera for the duration of a session.

    Writers are opened lazily on the first frame of each camera rather than at
    `start()`, because the frame size and the effective frame rate are only
    known once frames are actually arriving. A container's frame rate is fixed
    at open time and cannot be corrected later, so we take the viewer's measured
    display rate: guessing a nominal 30 fps for an ~18 fps stream would produce
    a file that plays back visibly fast.
    """

    # Frames buffered per camera to measure a frame rate before opening a
    # writer, when no estimate is available yet (e.g. --record from a cold start).
    PRIME_FRAMES = 15

    def __init__(self, outdir: Path | str, fourcc: str = "mp4v"):
        self.outdir = Path(outdir)
        self.fourcc = fourcc
        self.recording = False
        self._writers: dict[int, cv2.VideoWriter] = {}
        self._paths: dict[int, Path] = {}
        self._frames: dict[int, int] = {}
        self._pending: dict[int, list[tuple[np.ndarray, float]]] = {}
        self._session = ""
        self._started_at = 0.0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started_at if self.recording else 0.0

    @property
    def frame_count(self) -> int:
        return sum(self._frames.values())

    def start(self) -> None:
        if self.recording:
            return
        self.outdir.mkdir(parents=True, exist_ok=True)
        self._session = time.strftime("%Y%m%d-%H%M%S")
        self._started_at = time.monotonic()
        self._frames.clear()
        self._pending.clear()
        self.recording = True
        log.info("recording started -> %s/", self.outdir)

    def stop(self) -> list[Path]:
        """Close every writer and return the files produced."""
        if not self.recording:
            return []
        duration = self.elapsed
        for cam_id in list(self._pending):
            self._flush_pending(cam_id)
        self.recording = False
        for writer in self._writers.values():
            writer.release()

        written = []
        for cam_id, path in sorted(self._paths.items()):
            size_mb = path.stat().st_size / 1e6 if path.exists() else 0.0
            log.info(
                "cam%d: %d frames, %.1fs, %.1f MB -> %s",
                cam_id,
                self._frames.get(cam_id, 0),
                duration,
                size_mb,
                path,
            )
            written.append(path)

        self._writers.clear()
        self._paths.clear()
        return written

    def toggle(self) -> None:
        self.stop() if self.recording else self.start()

    def _open_writer(self, cam_id: int, frame: np.ndarray, fps: float) -> bool:
        height, width = frame.shape[:2]
        path = self.outdir / f"cam{cam_id}_{self._session}.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*self.fourcc), fps, (width, height)
        )
        if not writer.isOpened():
            log.error(
                "could not open %s with fourcc %r -- recording disabled. "
                "Try --record-fourcc avc1 (or MJPG with a .avi extension).",
                path,
                self.fourcc,
            )
            self.recording = False
            return False
        self._writers[cam_id] = writer
        self._paths[cam_id] = path
        self._frames[cam_id] = 0
        log.info("cam%d recording at %.1f fps -> %s", cam_id, fps, path)
        return True

    def _flush_pending(self, cam_id: int) -> None:
        """Open a writer using the rate measured across buffered frames, then
        write them out in order."""
        buffered = self._pending.pop(cam_id, [])
        if not buffered:
            return

        # Intervals come from the publisher's capture timestamps, never from
        # local arrival times: a subscriber joining mid-stream is handed its
        # whole backlog as one instant burst, so arrival times can report
        # hundreds of fps for a stream actually running at twenty. Capture
        # timestamps also stay correct when frames are dropped -- the recorded
        # interval genuinely is longer, and the file should play back that way.
        # The median absorbs any residual jitter.
        intervals = sorted(
            b[1] - a[1] for a, b in zip(buffered, buffered[1:]) if b[1] > a[1]
        )
        if intervals:
            mid = intervals[len(intervals) // 2]
            fps = 1.0 / mid if mid > 0 else 0.0
        else:
            fps = 0.0
        if not 1.0 < fps < 120.0:
            fps = 15.0
            log.warning("cam%d: could not measure a frame rate, defaulting to %.0f fps", cam_id, fps)

        if self._open_writer(cam_id, buffered[0][0], fps):
            writer = self._writers[cam_id]
            for frame, _ts in buffered:
                writer.write(frame)
            self._frames[cam_id] = len(buffered)

    def write(
        self, cam_id: int, frame: np.ndarray, fps: float, capture_ts: float | None = None
    ) -> None:
        """Record one frame. `capture_ts` is the publisher-side capture time
        (`FrameHeader.ts`); supply it whenever available, as it is the only
        reliable basis for the file's frame rate."""
        if not self.recording:
            return

        writer = self._writers.get(cam_id)
        if writer is None:
            if cam_id not in self._pending and 1.0 < fps < 120.0:
                # The viewer already has a solid estimate (the usual case: the
                # stream was on screen before RECORD was pressed).
                if not self._open_writer(cam_id, frame, fps):
                    return
                writer = self._writers[cam_id]
            else:
                # Cold start -- buffer briefly and measure the rate ourselves
                # rather than baking in a guess the container can never correct.
                pending = self._pending.setdefault(cam_id, [])
                pending.append((frame.copy(), capture_ts if capture_ts else time.monotonic()))
                if len(pending) >= self.PRIME_FRAMES:
                    self._flush_pending(cam_id)
                return

        writer.write(frame)
        self._frames[cam_id] = self._frames.get(cam_id, 0) + 1
