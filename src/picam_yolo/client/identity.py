"""Name the dogs the detector found, without stalling the preview.

The Pi says "dog"; this says *which* dog, by cropping the box the server already
published, embedding the crop, and matching it against an enrolled gallery. It
runs here rather than on the Pi for the reason `dogid` exists at all: the client
already has the JPEG and the boxes, so `protocol.py` is untouched and the board
-- which browns out at four cores -- does no extra work.

**Identification is far slower than the frame interval, and that shapes
everything here.** Measured on this dev machine, `mobilenet_v3_small` costs
~97 ms for a single crop and ~195 ms for four, against the ~52 ms between frames
at 19 fps. Running it inline would drop the preview to a stutter, so:

- A worker thread does the embedding. `submit()` never blocks and keeps only the
  *newest* pending frame, discarding anything queued behind it -- the same
  drop-don't-queue rule the viewer's `_drain()` and the server's PUB socket
  follow. A backlog here would mean naming a dog from a frame that is seconds
  old.
- The render thread draws whatever names the worker produced last, mapped onto
  the *current* boxes by IoU (`IdentityTracker`). Identity is stable over a few
  hundred milliseconds in a way that box position is not, so a name computed two
  frames ago is still the right answer for the box it has drifted into.

The pure logic (`identify_frame`, `IdentityTracker`) is deliberately separated
from the thread wrapper so both can be tested with `HashEmbedder`, no torch, no
camera and no Pi.
"""

from __future__ import annotations

import logging
import threading
import time
import zlib

import numpy as np

from ..protocol import Detection, FrameHeader

log = logging.getLogger(__name__)

# Carry a name across frames while the box still overlaps this much. Boxes move
# between identify runs; at 19 fps and ~200 ms per run a walking dog shifts a
# fraction of its own width, so this is loose on purpose.
IOU_MATCH = 0.3

# Drop results older than this. Without it, a name outlives the dog that earned
# it and lands on whatever walks through the same patch of frame next.
MAX_AGE_S = 2.0

DEFAULT_CLASSES = ("dog",)


def name_color(name: str) -> tuple[int, int, int]:
    """Stable, well-separated BGR colour per dog name.

    crc32 rather than hash(): PYTHONHASHSEED randomises str hashing per process,
    which would repaint every dog a different colour on each launch.
    """
    import cv2

    hue = zlib.crc32(name.encode()) % 180
    b, g, r = cv2.cvtColor(np.uint8([[[hue, 190, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def iou(a, b) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def identify_frame(
    frame_bgr: np.ndarray,
    header: FrameHeader,
    embedder,
    gallery,
    classes=DEFAULT_CLASSES,
    min_conf: float = 0.4,
) -> list[tuple[tuple[float, float, float, float], object]]:
    """Embed every qualifying crop in one batch and match it. Pure function.

    Crops are cut exactly as `dogid.capture` cuts them -- same `pad_box`, same
    `PAD_FRAC`, same `MIN_CROP_PX` -- because the gallery's centroids were built
    from crops of that shape. A tighter or looser crop at query time shifts the
    embedding distribution away from the one that was enrolled, which shows up
    as mysteriously low similarity rather than as an error.

    (The enrolled crops did make one extra q92 JPEG round trip on their way to
    disk that these do not. That loss is well below what the embedder resolves.)
    """
    from ..dogid.capture import MIN_CROP_PX, pad_box

    h, w = frame_bgr.shape[:2]
    boxes, crops = [], []
    for det in header.detections:
        if det.name not in classes or det.conf < min_conf:
            continue
        x1, y1, x2, y2 = pad_box(det.box, w, h)
        if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
            continue
        boxes.append(tuple(float(v) for v in det.box))
        crops.append(frame_bgr[y1:y2, x1:x2])

    if not crops:
        return []
    # One batched call: 4 crops cost ~195 ms where 4 separate calls cost ~390.
    return list(zip(boxes, gallery.match_batch(embedder.embed(crops))))


class IdentityTracker:
    """Holds the last identification and maps it onto the boxes on screen now."""

    def __init__(self, iou_match: float = IOU_MATCH, max_age: float = MAX_AGE_S):
        self.iou_match = iou_match
        self.max_age = max_age
        self._results: list[tuple[tuple, object]] = []
        self._at = 0.0

    def update(self, results, now: float | None = None) -> None:
        self._results = list(results)
        self._at = time.monotonic() if now is None else now

    def assign(self, header: FrameHeader, now: float | None = None) -> dict[int, object]:
        """Map stored matches onto `header.detections`, keyed by detection index.

        Greedy best-IoU rather than Hungarian: with a handful of dogs in frame
        the assignments are rarely contested, and a mislabelled frame costs a
        wrong name for ~200 ms, not a wrong training example.
        """
        now = time.monotonic() if now is None else now
        if not self._results or (now - self._at) > self.max_age:
            return {}

        out: dict[int, object] = {}
        taken: set[int] = set()
        for i, det in enumerate(header.detections):
            best, best_iou = None, self.iou_match
            for j, (box, match) in enumerate(self._results):
                if j in taken:
                    continue
                overlap = iou(det.box, box)
                if overlap >= best_iou:
                    best, best_iou = j, overlap
            if best is not None:
                taken.add(best)
                out[i] = self._results[best][1]
        return out


class DogIdentifier:
    """Runs `identify_frame` on a worker thread, newest frame wins.

    Start it, `submit()` every decoded frame, and read `labels()` when drawing.
    Neither call blocks the render loop.
    """

    def __init__(
        self,
        gallery,
        embedder,
        classes=DEFAULT_CLASSES,
        min_conf: float = 0.4,
        min_interval: float = 0.3,
    ):
        self.gallery = gallery
        self.embedder = embedder
        self.classes = tuple(classes)
        self.min_conf = min_conf
        # Cap how often the worker runs. Identification is the most expensive
        # thing this process does and a dog's identity does not change at 19 Hz.
        self.min_interval = min_interval

        # One tracker and one pending slot per camera. A single shared result
        # set would let cam0's dogs claim cam1's boxes -- the IoU test compares
        # coordinates, which say nothing about which camera they came from.
        self.trackers: dict[int, IdentityTracker] = {}
        self.last_ms = 0.0
        self._pending: dict[int, tuple[np.ndarray, FrameHeader]] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="dogid", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def submit(self, frame_bgr: np.ndarray, header: FrameHeader) -> None:
        """Offer a frame. Replaces any frame still waiting; never blocks."""
        if not any(d.name in self.classes for d in header.detections):
            return
        # The render loop draws onto its frame after handing it here, so the
        # worker must own a copy or it would embed a crop with a box painted
        # across the dog.
        with self._lock:
            self._pending[header.cam_id] = (frame_bgr.copy(), header)
        self._wake.set()

    def labels(self, header: FrameHeader) -> dict[int, object]:
        tracker = self.trackers.get(header.cam_id)
        return tracker.assign(header) if tracker else {}

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            if self._stop.is_set():
                return

            with self._lock:
                jobs, self._pending = self._pending, {}
            if not jobs:
                continue

            t0 = time.monotonic()
            for cam_id, (frame, header) in sorted(jobs.items()):
                try:
                    results = identify_frame(
                        frame, header, self.embedder, self.gallery,
                        classes=self.classes, min_conf=self.min_conf,
                    )
                except Exception:
                    # A failure here must not take the viewer down with it; the
                    # preview is still useful with no names on it.
                    log.exception("identification failed; continuing without names")
                    continue
                self.trackers.setdefault(cam_id, IdentityTracker()).update(results)
            self.last_ms = (time.monotonic() - t0) * 1000

            # Sleep off the remainder of the interval rather than spinning on
            # whatever the render loop has submitted in the meantime.
            slack = self.min_interval - (time.monotonic() - t0)
            if slack > 0:
                self._stop.wait(slack)


def create_identifier(
    gallery_path,
    backend: str | None = None,
    arch: str | None = None,
    weights: str | None = None,
    min_similarity: float | None = None,
    min_margin: float | None = None,
    **kwargs,
) -> DogIdentifier:
    """Build an identifier from a saved gallery.

    `dogid` is imported here rather than at module scope so that a client
    installed with only `[client]` -- no torch -- keeps working. Nothing on the
    viewing path touches this unless `--gallery` was passed.

    The backbone defaults to **whatever the gallery was built with**, matching
    `dogid`'s own default. Centroids mean nothing under another backbone, and a
    hardcoded default here meant that enrolling under a different `--arch` left
    the viewer silently mismatched -- it surfaced from a worker thread as a
    dimension error, not as advice.
    """
    from ..dogid.embed import create_embedder
    from ..dogid.gallery import Gallery

    gallery = Gallery.load(gallery_path)
    spec = gallery.embedder or {}
    if backend is None:
        backend = gallery.backend
    if arch is None:
        arch = spec.get("arch") or "mobilenet_v3_small"
    if weights is None:
        weights = spec.get("weights")
    # Thresholds are baked into the npz at enrol time, but they are also the
    # knob most worth turning while watching a live stream -- and re-enrolling
    # to try a value means re-embedding the whole dataset.
    if min_similarity is not None:
        gallery.min_similarity = min_similarity
    if min_margin is not None:
        gallery.min_margin = min_margin

    embed_kwargs = {"arch": arch, "weights_path": weights} if backend == "torch" else {}
    embedder = create_embedder(backend, **embed_kwargs)
    gallery.check_embedder(embedder)
    log.info(
        "identifying %d dog(s) [%s] with %s/%s: min_similarity=%.2f min_margin=%.2f",
        len(gallery.dog_names), ", ".join(gallery.dog_names), backend, arch,
        gallery.min_similarity, gallery.min_margin,
    )
    return DogIdentifier(gallery, embedder, **kwargs)
