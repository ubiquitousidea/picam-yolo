"""Object-detection backends.

Only NCNN-on-CPU is implemented today, but everything downstream talks to the
`Detector` protocol, so adding a Hailo or ONNX backend means adding a class here
and a branch in `create_detector` -- nothing in the pipeline changes.

Thread-safety: Ultralytics models are *not* safe to share across threads. Build
one detector per camera pipeline; the nano models are a few MB, so the duplicate
weights cost far less than the contention would.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol

import numpy as np

from ..protocol import Detection

log = logging.getLogger(__name__)


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...


class NullDetector:
    """Passes frames through undetected -- useful for isolating capture/network
    problems from inference problems."""

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        return []


class NcnnYoloDetector:
    """Ultralytics YOLO running on the NCNN backend.

    `model_path` must point at an exported ``*_ncnn_model`` directory (see
    scripts/export_model.py). `imgsz` has to match the size the model was
    exported at -- NCNN bakes the input dimensions in, and a mismatch produces
    silently wrong boxes rather than an error.
    """

    def __init__(
        self,
        model_path: str | Path,
        imgsz: int = 416,
        conf: float = 0.35,
        iou: float = 0.45,
        classes: list[int] | None = None,
        num_threads: int | None = None,
    ):
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.classes = classes

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"NCNN model not found at {self.model_path}. "
                "Run scripts/export_model.py to create it."
            )
        self._verify_model_files()

        # Cap intra-op parallelism before Ultralytics pulls in torch/cv2. With
        # two cameras on four cores, letting each backend grab every core makes
        # both pipelines slower than pinning them.
        if num_threads:
            os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
            os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))

        from ultralytics import YOLO

        self._model = YOLO(str(self.model_path), task="detect")
        log.info("loaded NCNN model %s at imgsz=%d", self.model_path.name, imgsz)

    def _verify_model_files(self) -> None:
        """Fail early and legibly on a truncated export.

        An unclean shutdown (this Pi browns out under load) leaves ext4 replaying
        its journal with recently written files zeroed. A 0-byte .param makes
        NCNN's load silently produce a graph with no input blobs, and Ultralytics
        then dies with `IndexError: list index out of range` deep inside its
        backend -- which the pipeline catches, so the server keeps reporting a
        healthy frame rate with zero detections forever. Checking here converts
        that into one clear message at startup.
        """
        for pattern, label in (("*.param", "graph"), ("*.bin", "weights")):
            files = list(self.model_path.glob(pattern))
            if not files:
                raise RuntimeError(
                    f"{self.model_path} contains no {pattern} file ({label}). "
                    "Re-run scripts/export_model.py."
                )
            for f in files:
                if f.stat().st_size == 0:
                    raise RuntimeError(
                        f"{f} is empty -- the export is truncated, most likely from an "
                        "unclean shutdown. Re-run scripts/export_model.py, or restore "
                        "from a known-good copy."
                    )

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names
        out: list[Detection] = []
        # Ultralytics undoes its own letterbox, so these are already in
        # frame_bgr pixel coordinates -- which is what the wire format wants.
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = (round(float(v), 1) for v in box.xyxy[0].tolist())
            out.append(
                Detection(
                    cls_id=cls_id,
                    name=names.get(cls_id, str(cls_id)),
                    conf=round(float(box.conf[0]), 3),
                    box=(x1, y1, x2, y2),
                )
            )
        return out


def create_detector(backend: str, **kwargs) -> Detector:
    """Factory keyed by backend name, so the CLI stays backend-agnostic."""
    if backend == "none":
        return NullDetector()
    if backend == "ncnn":
        return NcnnYoloDetector(**kwargs)
    raise ValueError(f"unknown detector backend: {backend!r}")
