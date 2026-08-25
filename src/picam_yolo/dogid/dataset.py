"""On-disk store of labelled dog crops.

Layout under the dataset root::

    crops/<sha1>.jpg     one cropped dog, content-addressed
    manifest.jsonl       one JSON record per crop, append-only

Two choices here are load-bearing.

**Content-addressed filenames.** Capture runs are cheap to repeat and a stream
happily hands you the same dog in near-identical frames. Naming a crop by the
SHA-1 of its JPEG bytes makes re-capture idempotent: the same image lands on
the same path and is skipped, so the dataset does not silently fill with
duplicates that would then bias the gallery toward whatever the camera saw most.

**Append-only JSONL rather than a rewritten JSON blob.** Labelling is a long
interactive session, and the dev machine is not immune to being closed mid-way.
Appending one line per change means a crash costs at most the last record, and
never the whole manifest. `load()` folds the log so later records win --
relabelling is just another append.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Reserved labels. UNKNOWN means "a dog, but not one we are tracking";
# NOT_A_DOG means the detector was wrong and the crop is not a dog at all.
# Both are kept rather than deleted: they are the negatives that stop the
# gallery from confidently matching a cat, a rug, or the neighbour's collie.
UNLABELLED = ""
UNKNOWN = "__unknown__"
NOT_A_DOG = "__not_a_dog__"
RESERVED = {UNKNOWN, NOT_A_DOG}


@dataclass
class CropRecord:
    """One dog crop and what we know about it."""

    crop_id: str  # sha1 of the JPEG bytes; also the filename stem
    source: str  # where it came from, e.g. "cam0" or a video path
    ts: float  # capture time, epoch seconds, from FrameHeader.ts
    box: tuple[float, float, float, float]  # in the *source frame*, for provenance
    det_conf: float  # the detector's confidence that this was a dog
    label: str = UNLABELLED
    split: str = "train"  # train | val, assigned at label time
    labelled_at: float = 0.0

    @property
    def is_labelled(self) -> bool:
        return self.label != UNLABELLED

    @property
    def is_identity(self) -> bool:
        """True for a real dog name -- the only records that seed the gallery."""
        return self.is_labelled and self.label not in RESERVED


@dataclass
class CropDataset:
    """Crops plus manifest, rooted at a directory."""

    root: Path
    records: dict[str, CropRecord] = field(default_factory=dict)

    @property
    def crops_dir(self) -> Path:
        return self.root / "crops"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.jsonl"

    @classmethod
    def open(cls, root: Path | str) -> CropDataset:
        root = Path(root)
        ds = cls(root=root)
        (root / "crops").mkdir(parents=True, exist_ok=True)
        if ds.manifest.exists():
            ds._load()
        return ds

    def _load(self) -> None:
        """Fold the append-only log; later records for a crop_id win."""
        bad = 0
        for line in self.manifest.read_text().splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                data["box"] = tuple(data["box"])
                rec = CropRecord(**data)
            except (json.JSONDecodeError, TypeError, KeyError):
                # A truncated final line is the expected failure after an
                # interrupted session. Skip it rather than refusing to open.
                bad += 1
                continue
            self.records[rec.crop_id] = rec
        if bad:
            log.warning("skipped %d unreadable manifest line(s)", bad)

    def _append(self, rec: CropRecord) -> None:
        with self.manifest.open("a") as fh:
            fh.write(json.dumps(asdict(rec), separators=(",", ":")) + "\n")
        self.records[rec.crop_id] = rec

    def add(self, jpeg: bytes, source: str, ts: float, box, det_conf: float) -> str | None:
        """Store a crop. Returns its id, or None if it was already present."""
        crop_id = hashlib.sha1(jpeg).hexdigest()
        if crop_id in self.records:
            return None
        (self.crops_dir / f"{crop_id}.jpg").write_bytes(jpeg)
        self._append(
            CropRecord(
                crop_id=crop_id,
                source=source,
                ts=ts,
                box=tuple(float(v) for v in box),
                det_conf=float(det_conf),
            )
        )
        return crop_id

    def label(self, crop_id: str, label: str, split: str = "train") -> None:
        rec = self.records[crop_id]
        self._append(
            CropRecord(
                **{
                    **asdict(rec),
                    "box": tuple(rec.box),
                    "label": label,
                    "split": split,
                    "labelled_at": time.time(),
                }
            )
        )

    def path_for(self, crop_id: str) -> Path:
        return self.crops_dir / f"{crop_id}.jpg"

    def image(self, crop_id: str) -> np.ndarray:
        """Decode one crop to BGR. Imported lazily so that capture and stats
        work on a machine without OpenCV."""
        import cv2

        img = cv2.imread(str(self.path_for(crop_id)), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"crop {crop_id} missing or unreadable")
        return img

    # -- queries -----------------------------------------------------------

    def unlabelled(self) -> list[CropRecord]:
        return [r for r in self.records.values() if not r.is_labelled]

    def identities(self, split: str | None = None) -> list[CropRecord]:
        return [
            r
            for r in self.records.values()
            if r.is_identity and (split is None or r.split == split)
        ]

    def negatives(self) -> list[CropRecord]:
        return [r for r in self.records.values() if r.label in RESERVED]

    def names(self) -> list[str]:
        return sorted({r.label for r in self.records.values() if r.is_identity})

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records.values():
            key = r.label or "(unlabelled)"
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))
