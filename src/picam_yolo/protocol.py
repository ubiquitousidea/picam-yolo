"""Wire format shared by the Pi server and the desktop client.

Each frame travels as a three-part ZeroMQ message::

    [ topic ] [ header JSON ] [ JPEG bytes ]

``topic`` is ``b"cam0"``, ``b"cam1"``, ... so a subscriber can filter to a single
camera with a prefix subscription instead of decoding everything.

Detection boxes are stored in pixel coordinates *of the JPEG payload*. The server
resolves the inference letterbox back to display pixels before publishing, so the
client never needs to know the model's input size. This is the only contract
between the two halves -- keep them in sync when changing it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

WIRE_VERSION = 1


def topic_for(cam_id: int) -> bytes:
    """ZeroMQ topic for a camera index. Prefix-matched, so keep it fixed-width-ish."""
    return f"cam{cam_id}".encode()


@dataclass(frozen=True)
class Detection:
    """One YOLO box, in JPEG-payload pixel coordinates."""

    cls_id: int
    name: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class FrameHeader:
    cam_id: int
    seq: int
    ts: float  # capture time, epoch seconds
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)
    capture_ms: float = 0.0
    infer_ms: float = 0.0
    encode_ms: float = 0.0
    version: int = WIRE_VERSION

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, raw: bytes) -> FrameHeader:
        data = json.loads(raw)
        dets = [
            Detection(
                cls_id=d["cls_id"],
                name=d["name"],
                conf=d["conf"],
                box=tuple(d["box"]),  # type: ignore[arg-type]
            )
            for d in data.pop("detections", [])
        ]
        return cls(detections=dets, **data)
