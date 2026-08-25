"""Subscribe to the Pi's frame stream and draw it in native OpenCV windows."""

from __future__ import annotations

import logging
import time
from collections import deque

import cv2
import numpy as np
import zmq

from ..protocol import FrameHeader
from .recorder import StreamRecorder

log = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Record button geometry, in frame pixel coordinates, anchored to the top right.
_BTN_W, _BTN_H, _BTN_PAD = 132, 34, 12


def record_button_rect(frame_width: int) -> tuple[int, int, int, int]:
    """Button bounds as (x1, y1, x2, y2). Single source of truth so that drawing
    and hit-testing can never drift apart."""
    x2 = frame_width - _BTN_PAD
    return x2 - _BTN_W, _BTN_PAD, x2, _BTN_PAD + _BTN_H


def class_color(cls_id: int) -> tuple[int, int, int]:
    """Stable, well-separated BGR colour per class id."""
    hue = (cls_id * 47) % 180  # 47 is coprime with 180, so ids spread out
    hsv = np.uint8([[[hue, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def draw_overlay(frame: np.ndarray, header: FrameHeader, fps: float, show_hud: bool) -> np.ndarray:
    for det in header.detections:
        x1, y1, x2, y2 = (int(v) for v in det.box)
        color = class_color(det.cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{det.name} {det.conf:.2f}"
        (tw, th), base = cv2.getTextSize(label, _FONT, 0.5, 1)
        # Keep the label inside the frame when the box hugs the top edge.
        top = max(y1, th + base + 2)
        cv2.rectangle(frame, (x1, top - th - base - 2), (x1 + tw + 4, top), color, -1)
        cv2.putText(frame, label, (x1 + 2, top - base), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if show_hud:
        # Depends on the Pi and this machine agreeing on the time; both run NTP,
        # so treat it as indicative rather than a precise measurement.
        latency_ms = (time.time() - header.ts) * 1000
        lines = [
            f"cam{header.cam_id}  {header.width}x{header.height}  {fps:4.1f} fps",
            f"infer {header.infer_ms:5.1f}ms  encode {header.encode_ms:4.1f}ms  e2e {latency_ms:5.0f}ms",
            f"{len(header.detections)} object(s)",
        ]
        for i, line in enumerate(lines):
            y = 22 + i * 20
            cv2.putText(frame, line, (10, y), _FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (10, y), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_record_button(frame: np.ndarray, recording: bool, elapsed: float) -> None:
    """Draw the record control. Called after the frame has been handed to the
    recorder, so this chrome never appears in the saved video."""
    x1, y1, x2, y2 = record_button_rect(frame.shape[1])

    # Dim plate behind the control so it stays readable over a bright scene.
    plate = frame[y1:y2, x1:x2]
    if plate.size:
        frame[y1:y2, x1:x2] = (plate * 0.35).astype(np.uint8)

    accent = (60, 60, 235) if recording else (210, 210, 210)
    cv2.rectangle(frame, (x1, y1), (x2, y2), accent, 1)

    cy = (y1 + y2) // 2
    if recording:
        cv2.circle(frame, (x1 + 18, cy), 7, accent, -1)
        label = f"REC {int(elapsed) // 60:01d}:{int(elapsed) % 60:02d}"
    else:
        cv2.circle(frame, (x1 + 18, cy), 7, accent, 2)
        label = "RECORD"
    cv2.putText(frame, label, (x1 + 34, cy + 5), _FONT, 0.5, accent, 1, cv2.LINE_AA)


class Viewer:
    """Drains the socket to the newest frame per camera before drawing.

    ZeroMQ will happily hand over a queued backlog; showing it would mean
    watching the past. Reading everything available and keeping only the latest
    frame per camera keeps the window live when the Pi outruns the link.
    """

    def __init__(
        self,
        endpoint: str,
        cameras: list[int] | None = None,
        rcvhwm: int = 4,
        recorder: StreamRecorder | None = None,
        record_raw: bool = False,
    ):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, rcvhwm)
        self._sock.connect(endpoint)

        if cameras:
            for cam in cameras:
                self._sock.setsockopt(zmq.SUBSCRIBE, f"cam{cam}".encode())
        else:
            self._sock.setsockopt(zmq.SUBSCRIBE, b"")

        self._endpoint = endpoint
        self._windows: set[str] = set()
        self._frame_times: dict[int, deque] = {}
        self.show_hud = True
        self.recorder = recorder or StreamRecorder("recordings")
        self.record_raw = record_raw

    def _fps(self, cam_id: int) -> float:
        times = self._frame_times.setdefault(cam_id, deque(maxlen=30))
        times.append(time.monotonic())
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0 else 0.0

    def _on_mouse(self, event: int, x: int, y: int, flags: int, frame_width) -> None:
        """Toggle recording when the button is clicked.

        Coordinates arrive in image space, so the hit test matches the drawn
        rectangle exactly at the window's native size. Resizing the window can
        offset it on some OpenCV backends -- the `r` key is the reliable path.
        """
        if event != cv2.EVENT_LBUTTONDOWN or not frame_width:
            return
        x1, y1, x2, y2 = record_button_rect(frame_width)
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.recorder.toggle()

    def _drain(self, timeout_ms: int) -> dict[int, tuple[FrameHeader, bytes]]:
        """Block up to `timeout_ms` for traffic, then take everything queued."""
        latest: dict[int, tuple[FrameHeader, bytes]] = {}
        if not self._sock.poll(timeout_ms):
            return latest
        while True:
            try:
                _topic, raw_header, jpeg = self._sock.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return latest
            except ValueError:
                log.warning("malformed message dropped")
                continue
            header = FrameHeader.from_bytes(raw_header)
            latest[header.cam_id] = (header, jpeg)

    def run(self) -> int:
        log.info(
            "connected to %s -- click RECORD or press r to record, h toggles the HUD, q quits",
            self._endpoint,
        )
        waiting_logged = False

        try:
            while True:
                frames = self._drain(timeout_ms=200)
                if not frames and not self._windows and not waiting_logged:
                    log.info("waiting for frames from %s ...", self._endpoint)
                    waiting_logged = True

                for cam_id, (header, jpeg) in sorted(frames.items()):
                    buf = np.frombuffer(jpeg, dtype=np.uint8)
                    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if frame is None:
                        log.warning("cam%d: undecodable JPEG dropped", cam_id)
                        continue

                    title = f"cam{cam_id}"
                    if title not in self._windows:
                        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(title, header.width, header.height)
                        cv2.setMouseCallback(title, self._on_mouse, header.width)
                        self._windows.add(title)

                    fps = self._fps(cam_id)
                    if self.record_raw:
                        # Capture the clean frame before any annotation lands on it.
                        self.recorder.write(cam_id, frame, fps, header.ts)

                    draw_overlay(frame, header, fps, self.show_hud)
                    if not self.record_raw:
                        self.recorder.write(cam_id, frame, fps, header.ts)

                    draw_record_button(frame, self.recorder.recording, self.recorder.elapsed)
                    cv2.imshow(title, frame)

                # waitKey also pumps the window event loop, so it must run every
                # iteration even when no frame arrived.
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("h"):
                    self.show_hud = not self.show_hud
                if key == ord("r"):
                    self.recorder.toggle()

                # Quit if the user closed the last window with the title-bar button.
                if self._windows and all(
                    cv2.getWindowProperty(w, cv2.WND_PROP_VISIBLE) < 1 for w in self._windows
                ):
                    break
        finally:
            # Never leave a half-written container behind on exit or Ctrl-C.
            written = self.recorder.stop()
            if written:
                log.info("saved %d recording(s)", len(written))
            cv2.destroyAllWindows()
            self._sock.close(linger=0)
        return 0
