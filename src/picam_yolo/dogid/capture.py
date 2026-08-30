"""Harvest dog crops from the live stream.

The Pi has *already* run YOLO on every frame it publishes, and the boxes travel
in `FrameHeader`. So harvesting training data needs no model on this machine and
costs the Pi nothing -- subscribe, keep the frames with a dog in them, cut out
the box, write the crop. That is the whole idea.

Two details that matter for data quality rather than correctness:

**Padding.** A tight detector box clips ears, tails and muzzle -- exactly the
fine detail that distinguishes one dog from another. Crops are expanded by
`PAD_FRAC` on each side before saving.

**Novelty gating.** A dog asleep in front of the camera produces hundreds of
near-identical frames. Feeding all of them in makes the gallery confident about
one pose and useless at any other, and makes labelling tedious for no gain. A
cheap perceptual hash drops crops that closely resemble one just kept.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..protocol import FrameHeader
from .dataset import CropDataset

log = logging.getLogger(__name__)

PAD_FRAC = 0.12
MIN_CROP_PX = 64  # below this there is not enough detail to identify anyone
# The classes a dog gets read as. Harvesting only "dog" throws away exactly the
# frames the gallery most needs: a dog the detector called a cat is a hard pose,
# and it is also the frame we would want labelled __not_a_dog__ if it really was
# a cat. Either way the crop is worth keeping -- the gallery's rejection class,
# not the detector, is what decides.
#
# `client.identity` keeps its own copy of this tuple rather than importing it:
# that module must stay importable with only the [client] extra installed, and a
# module-scope import of dogid would break that. Keep the two in step.
DEFAULT_CLASSES = ("dog", "cat", "teddy bear")


def pad_box(box, width: int, height: int, frac: float = PAD_FRAC):
    """Expand a box by at least `frac` on each side, clipped to the frame.

    Floor the minimums and ceil the maximums. Truncating both ends with int()
    instead would expand the left edge while *contracting* the right, quietly
    shifting every crop a fraction of a box to the left -- a systematic bias in
    training data is a worse bug than a slightly loose crop.
    """
    import math

    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * frac, (y2 - y1) * frac
    return (
        max(0, math.floor(x1 - dx)),
        max(0, math.floor(y1 - dy)),
        min(width, math.ceil(x2 + dx)),
        min(height, math.ceil(y2 + dy)),
    )


def dhash(img_bgr: np.ndarray, size: int = 8) -> int:
    """64-bit difference hash. Cheap, and good enough to spot 'the same dog in
    the same pose half a second later', which is all we need."""
    import cv2

    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = g[:, 1:] > g[:, :-1]
    out = 0
    for bit in bits.flatten():
        out = (out << 1) | int(bit)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class CropHarvester:
    """Turns (frame, header) pairs into dataset entries.

    Kept separate from the ZeroMQ loop so it can be driven from a recorded video
    or a test just as easily as from the live stream.
    """

    def __init__(
        self,
        dataset: CropDataset,
        classes=DEFAULT_CLASSES,
        min_conf: float = 0.4,
        novelty: int = 8,
        jpeg_quality: int = 92,
    ):
        self.dataset = dataset
        self.classes = set(classes)
        self.min_conf = min_conf
        self.novelty = novelty  # min hamming distance from a recently kept crop
        self.jpeg_quality = jpeg_quality
        self._recent: list[int] = []
        self.kept = 0
        self.skipped_dup = 0
        self.skipped_small = 0

    def feed(self, frame_bgr: np.ndarray, header: FrameHeader) -> int:
        """Harvest from one frame. Returns the number of crops kept."""
        import cv2

        kept = 0
        h, w = frame_bgr.shape[:2]
        for det in header.detections:
            if det.name not in self.classes or det.conf < self.min_conf:
                continue
            x1, y1, x2, y2 = pad_box(det.box, w, h)
            if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
                self.skipped_small += 1
                continue

            crop = frame_bgr[y1:y2, x1:x2]
            digest = dhash(crop)
            if any(hamming(digest, seen) < self.novelty for seen in self._recent):
                self.skipped_dup += 1
                continue

            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                continue
            added = self.dataset.add(
                buf.tobytes(),
                source=f"cam{header.cam_id}",
                ts=header.ts,
                box=det.box,
                det_conf=det.conf,
                det_name=det.name,
            )
            if added is None:
                self.skipped_dup += 1
                continue

            # Bounded history: only recent frames can plausibly be near-duplicates,
            # and an unbounded list would make this O(n) per crop over a long run.
            self._recent.append(digest)
            del self._recent[:-64]
            self.kept += 1
            kept += 1
        return kept


def harvest_stream(
    dataset: CropDataset,
    host: str,
    port: int,
    seconds: float,
    cameras=None,
    **kwargs,
) -> int:
    """Subscribe to the Pi and harvest for `seconds`. Returns crops kept."""
    import cv2
    import zmq

    from ..protocol import topic_for

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{host}:{port}")
    if cameras:
        for cam in cameras:
            sock.setsockopt(zmq.SUBSCRIBE, topic_for(cam))
    else:
        sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVTIMEO, 5000)

    harvester = CropHarvester(dataset, **kwargs)
    deadline = time.time() + seconds
    frames = 0
    try:
        while time.time() < deadline:
            try:
                _topic, raw, jpeg = sock.recv_multipart()
            except zmq.Again:
                log.warning("no frames for 5s -- is the server running?")
                continue
            header = FrameHeader.from_bytes(raw)
            frames += 1
            # Decode only when there is something worth cropping; at 19 fps the
            # decode is the expensive part of this loop.
            if not any(d.name in harvester.classes for d in header.detections):
                continue
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                harvester.feed(frame, header)
    finally:
        sock.close(linger=0)

    log.info(
        "saw %d frames, kept %d crops (%d near-duplicates, %d too small)",
        frames,
        harvester.kept,
        harvester.skipped_dup,
        harvester.skipped_small,
    )
    return harvester.kept
