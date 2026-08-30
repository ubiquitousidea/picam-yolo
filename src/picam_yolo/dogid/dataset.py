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

# Reserved labels, none of which name a dog.
#
# UNKNOWN means "a dog, but not one we are tracking"; NOT_A_DOG means the
# detector was wrong and this is not a dog at all. Both are kept rather than
# deleted: they are the negatives that stop the gallery from confidently
# matching a cat, a rug, or the neighbour's collie.
#
# DISCARD is different, and the distinction is load-bearing. It marks a crop
# that is unusable rather than negative -- two dogs inside one padded box, a
# motion-blurred smear, a tail leaving the frame. Such a crop must not seed a
# dog's centroid, but it must *equally* not seed the rejection class: a crop
# containing two dogs we know would teach the gallery that our own dogs look
# like `__reject__`, dragging that centroid toward the very thing it exists to
# distinguish. So `RESERVED` (not an identity) and `NEGATIVE_LABELS` (seeds the
# rejection class) are deliberately different sets.
#
# Skipping such a crop instead leaves it unlabelled, which is correct for the
# gallery but means `unlabelled()` offers it again every session. DISCARD is
# how a judgement gets recorded once.
UNLABELLED = ""
UNKNOWN = "__unknown__"
NOT_A_DOG = "__not_a_dog__"
DISCARD = "__discard__"
NEGATIVE_LABELS = {UNKNOWN, NOT_A_DOG}
RESERVED = NEGATIVE_LABELS | {DISCARD}


@dataclass
class CropRecord:
    """One dog crop and what we know about it."""

    crop_id: str  # sha1 of the JPEG bytes; also the filename stem
    source: str  # where it came from, e.g. "cam0" or a video path
    ts: float  # capture time, epoch seconds, from FrameHeader.ts
    box: tuple[float, float, float, float]  # in the *source frame*, for provenance
    det_conf: float  # the detector's confidence in `det_name`
    # What the detector actually called it. Crops are harvested from the classes
    # a dog gets confused for, not just "dog", so this is the difference between
    # "a clear dog" and "a dog the detector called a teddy bear" -- which is
    # exactly what you want to know while labelling. Defaulted, so records
    # written before this field existed load unchanged; "dog" is correct for
    # them, since nothing else was harvested then.
    det_name: str = "dog"
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

    def add(
        self, jpeg: bytes, source: str, ts: float, box, det_conf: float, det_name: str = "dog"
    ) -> str | None:
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
                det_name=det_name,
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

    # -- embeddings --------------------------------------------------------

    def embedding_cache(self, embedder) -> Path:
        """Where vectors for this embedder live.

        Keyed by backend, width *and* weights, because a fine-tuned checkpoint
        produces different vectors from the same architecture -- serving those
        from a cache named only after the arch would silently mix two embedding
        spaces, which is the one failure this cache must never cause.
        """
        spec = embedder.spec() if hasattr(embedder, "spec") else {}
        parts = [str(spec.get("backend", "unknown")), str(getattr(embedder, "dim", 0))]
        if spec.get("weights"):
            parts.append(Path(str(spec["weights"])).stem)
        return self.root / f"embeddings-{'-'.join(parts)}.npz"

    def embed_ids(self, embedder, crop_ids: list[str], batch: int = 32) -> np.ndarray:
        """Embed crops, reusing anything already computed for this embedder.

        Crops are content-addressed, so a crop_id always names the same pixels
        and a cached vector can never go stale. That makes this a pure win:
        ranking 1470 crops by gallery margin took ~120 s of torch every time
        `label` started, and every re-run of `enrol` or `eval` paid it again.

        Missing or unreadable crops are skipped, so the result can be shorter
        than `crop_ids`; callers that need the correspondence should use
        `embed_ids_present`.
        """
        vecs, _ = self.embed_ids_present(embedder, crop_ids, batch)
        return vecs

    def embed_ids_present(
        self, embedder, crop_ids: list[str], batch: int = 32
    ) -> tuple[np.ndarray, list[str]]:
        """As `embed_ids`, but also returns the ids that were actually embedded."""
        path = self.embedding_cache(embedder)
        cache: dict[str, np.ndarray] = {}
        if path.exists():
            try:
                data = np.load(path, allow_pickle=False)
                cache = dict(zip((str(i) for i in data["ids"]), data["vecs"]))
            except (OSError, ValueError, KeyError):
                # A cache is an optimisation; a corrupt one is not a reason to
                # refuse to work. Recompute and overwrite it.
                log.warning("ignoring unreadable embedding cache %s", path)

        missing = [cid for cid in crop_ids if cid not in cache]
        added = 0
        for i in range(0, len(missing), batch):
            imgs, kept = [], []
            for cid in missing[i : i + batch]:
                try:
                    imgs.append(self.image(cid))
                    kept.append(cid)
                except (FileNotFoundError, ValueError):
                    continue
            if not imgs:
                continue
            for cid, vec in zip(kept, embedder.embed(imgs)):
                cache[cid] = vec
                added += 1

        if added:
            path.parent.mkdir(parents=True, exist_ok=True)
            ids = list(cache)
            np.savez(path, ids=np.array(ids), vecs=np.stack([cache[i] for i in ids]))
            log.info("embedded %d new crop(s); cache now holds %d", added, len(cache))

        present = [cid for cid in crop_ids if cid in cache]
        if not present:
            return np.zeros((0, getattr(embedder, "dim", 0)), dtype=np.float32), []
        return np.stack([cache[cid] for cid in present]), present

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
        """Crops that seed the rejection class -- deliberately not all of
        RESERVED, since DISCARD marks unusable, not negative."""
        return [r for r in self.records.values() if r.label in NEGATIVE_LABELS]

    def discarded(self) -> list[CropRecord]:
        return [r for r in self.records.values() if r.label == DISCARD]

    def names(self) -> list[str]:
        return sorted({r.label for r in self.records.values() if r.is_identity})

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records.values():
            key = r.label or "(unlabelled)"
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))
