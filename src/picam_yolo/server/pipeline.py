"""Per-camera capture -> detect -> encode -> publish loop."""

from __future__ import annotations

import errno
import logging
import threading
import time

import numpy as np
import zmq

from ..protocol import FrameHeader, topic_for
from .cameras import FrameSource
from .detector import Detector

log = logging.getLogger(__name__)

# A detector that fails this many times running is broken, not unlucky.
MAX_CONSECUTIVE_DETECT_FAILURES = 15


def _make_encoder(quality: int):
    """Prefer simplejpeg (ships with picamera2, libjpeg-turbo backed); fall back
    to OpenCV so the server can also run on a dev machine."""
    try:
        import simplejpeg

        def encode(frame: np.ndarray) -> bytes:
            return simplejpeg.encode_jpeg(
                np.ascontiguousarray(frame), quality=quality, colorspace="BGR"
            )

        return encode
    except ImportError:
        import cv2

        def encode(frame: np.ndarray) -> bytes:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError("JPEG encode failed")
            return buf.tobytes()

        return encode


class FramePublisher:
    """Thread-safe PUB socket shared by every camera pipeline.

    A send HWM of a few frames is deliberate: when the client (or the network)
    can't keep up we want ZeroMQ to drop the backlog and keep the newest frames
    flowing, rather than queue up latency we can never pay off.
    """

    def __init__(self, bind_addr: str, hwm: int = 4):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, hwm)
        try:
            self._sock.bind(bind_addr)
        except zmq.ZMQError as exc:
            self._sock.close(linger=0)
            if exc.errno == errno.EADDRINUSE:
                # Almost always a server left over from a previous launch. A
                # bare traceback here sends people hunting the wrong problem.
                raise RuntimeError(
                    f"{bind_addr} is already in use -- another picam_yolo.server "
                    f"is probably still running. Stop it with:\n"
                    f"    pkill -f '[p]icam_yolo.server'\n"
                    f"(the [p] is required: a plain -f pattern matches, and kills, "
                    f"your own ssh session)"
                ) from exc
            raise
        self._lock = threading.Lock()
        log.info("publishing on %s", bind_addr)

    def publish(self, header: FrameHeader, jpeg: bytes) -> None:
        payload = [topic_for(header.cam_id), header.to_bytes(), jpeg]
        with self._lock:
            self._sock.send_multipart(payload, copy=False)

    def close(self) -> None:
        with self._lock:
            self._sock.close(linger=0)


class CameraPipeline(threading.Thread):
    """Owns one camera end to end and runs until `stop()`.

    `detect_every` > 1 decouples stream rate from inference rate: every frame is
    published, but detection only runs on every Nth one and the previous boxes
    are reused in between. On a CPU-only Pi that is the difference between a
    smooth 15 fps preview with slightly stale boxes and a stuttering 5 fps one.
    """

    def __init__(
        self,
        source: FrameSource,
        detector: Detector,
        publisher: FramePublisher,
        jpeg_quality: int = 80,
        detect_every: int = 1,
    ):
        super().__init__(name=f"pipeline-cam{source.info.num}", daemon=True)
        self.source = source
        self.detector = detector
        self.publisher = publisher
        self.detect_every = max(1, detect_every)
        self._encode = _make_encoder(jpeg_quality)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        cam_id = self.source.info.num
        seq = 0
        last_detections = []
        consecutive_failures = 0
        window_start = time.monotonic()
        window_frames = 0
        # Accumulate per-stage cost over the log window. Reporting the *last*
        # frame's timings instead is misleading whenever detect_every > 1: that
        # frame either ran inference or skipped it, so `infer` alternates
        # between the true cost and 0.0 and neither number is the answer.
        window_capture = window_encode = window_infer = 0.0
        window_inferred = 0  # frames that actually ran inference

        try:
            self.source.start()
        except Exception:
            log.exception("%s failed to start", self.source.info.label)
            return

        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                frame = self.source.read()
                t_cap = time.monotonic()

                inferred = seq % self.detect_every == 0
                if inferred:
                    try:
                        last_detections = self.detector.detect(frame)
                        consecutive_failures = 0
                    except Exception:
                        consecutive_failures += 1
                        # Tolerate the occasional transient, but never degrade
                        # silently forever: a permanently broken detector
                        # otherwise looks like a healthy stream with no objects
                        # in view, which is indistinguishable from working.
                        if consecutive_failures == 1:
                            log.exception("cam%d detection failed; publishing raw frames", cam_id)
                        if consecutive_failures >= MAX_CONSECUTIVE_DETECT_FAILURES:
                            log.error(
                                "cam%d: detection has failed %d times in a row -- stopping. "
                                "The stream was being published without any detections.",
                                cam_id,
                                consecutive_failures,
                            )
                            return
                        last_detections = []
                t_inf = time.monotonic()

                jpeg = self._encode(frame)
                t_enc = time.monotonic()

                height, width = frame.shape[:2]
                header = FrameHeader(
                    cam_id=cam_id,
                    seq=seq,
                    ts=time.time(),
                    width=width,
                    height=height,
                    detections=last_detections,
                    capture_ms=round((t_cap - t0) * 1000, 2),
                    infer_ms=round((t_inf - t_cap) * 1000, 2),
                    encode_ms=round((t_enc - t_inf) * 1000, 2),
                )
                self.publisher.publish(header, jpeg)

                seq += 1
                window_frames += 1
                window_capture += header.capture_ms
                window_encode += header.encode_ms
                # Averaged over inferring frames only, so the figure means "what
                # one inference costs" and stays comparable across detect_every
                # settings. Divide by window_frames instead and it silently
                # reports the amortised cost, which halves when detect_every
                # doubles even though the model has not changed.
                if inferred:
                    window_infer += header.infer_ms
                    window_inferred += 1
                elapsed = time.monotonic() - window_start
                if elapsed >= 5.0:
                    log.info(
                        "cam%d %.1f fps (capture %.1fms, infer %.1fms over %d frames, "
                        "encode %.1fms, %d boxes)",
                        cam_id,
                        window_frames / elapsed,
                        window_capture / window_frames,
                        window_infer / window_inferred if window_inferred else 0.0,
                        window_inferred,
                        window_encode / window_frames,
                        len(last_detections),
                    )
                    window_start, window_frames = time.monotonic(), 0
                    window_capture = window_encode = window_infer = 0.0
                    window_inferred = 0
        finally:
            self.source.stop()
            log.info("cam%d pipeline stopped after %d frames", cam_id, seq)
