"""Evaluate a gallery, and sharpen the embedder that feeds it.

The order matters. `evaluate` is complete and is the thing to run first: with a
pretrained backbone and no training at all, enrolling a handful of crops per dog
often already works, and the eval tells you whether it does. Only if the
confusion matrix shows two dogs bleeding into each other is `finetune` worth the
effort -- ImageNet features separate a spaniel from a labrador easily and two
black labradors barely at all.

`finetune` is a skeleton. The structure and the seams are here; the training
loop itself is marked TODO because the right loss depends on data that does not
exist yet (batch-hard triplet needs several crops per dog per batch, which needs
enough dogs to sample from).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dataset import CropDataset
from .embed import Embedder
from .gallery import Gallery

log = logging.getLogger(__name__)


@dataclass
class EvalReport:
    n: int = 0
    correct: int = 0
    rejected: int = 0  # true dog, but the gallery declined to name it
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.n if self.n else 0.0

    def render(self) -> str:
        lines = [
            f"n={self.n}  accuracy={self.accuracy:.1%}  "
            f"rejected={self.rejection_rate:.1%}  (of held-out crops of known dogs)",
            "",
            "confusion (row = truth, col = predicted):",
        ]
        cols = sorted({c for row in self.confusion.values() for c in row})
        if cols:
            width = max(len(c) for c in list(cols) + list(self.confusion)) + 2
            lines.append(" " * width + "".join(f"{c:>{width}}" for c in cols))
            for truth in sorted(self.confusion):
                row = self.confusion[truth]
                lines.append(
                    f"{truth:<{width}}" + "".join(f"{row.get(c, 0):>{width}}" for c in cols)
                )
        return "\n".join(lines)


def evaluate(dataset: CropDataset, embedder: Embedder, gallery: Gallery, split: str = "val") -> EvalReport:
    """Score the gallery on held-out crops.

    Rejections are counted separately from mistakes on purpose. Naming the wrong
    dog and declining to name a known one are different failures with different
    fixes -- the first wants a better embedder, the second usually just wants
    `min_similarity` lowered.
    """
    report = EvalReport()
    records = dataset.identities(split=split)
    if not records:
        log.warning("no %r crops -- label more, every %dth goes to validation", split, 5)
        return report

    by_id = {r.crop_id: r for r in records}
    vecs, ids = dataset.embed_ids_present(embedder, list(by_id))
    for cid, vec in zip(ids, vecs):
        rec = by_id[cid]
        predicted = gallery.match(vec).name
        report.n += 1
        key = predicted or "(rejected)"
        report.confusion.setdefault(rec.label, {})
        report.confusion[rec.label][key] = report.confusion[rec.label].get(key, 0) + 1
        if predicted == rec.label:
            report.correct += 1
        elif predicted is None:
            report.rejected += 1
    return report


def suggest_threshold(dataset: CropDataset, embedder: Embedder, gallery: Gallery) -> float:
    """Pick `min_similarity` from the data instead of guessing.

    Returns the similarity that best separates same-dog from different-dog
    pairs on the validation split. A hand-picked default is a coin flip: the
    right value depends on the backbone and on how visually alike these
    particular dogs are.
    """
    same, diff = [], []
    val = {r.crop_id: r for r in dataset.identities(split="val")}
    vecs, ids = dataset.embed_ids_present(embedder, list(val))
    for cid, vec in zip(ids, vecs):
        rec = val[cid]
        sims = gallery.centroids @ (vec / max(np.linalg.norm(vec), 1e-9))
        for name, sim in zip(gallery.names, sims):
            (same if name == rec.label else diff).append(float(sim))

    if not same or not diff:
        log.warning("not enough validation data to suggest a threshold")
        return gallery.min_similarity

    # Midpoint between the distributions, biased toward the negatives so a
    # stranger is rejected rather than misnamed.
    cut = (float(np.percentile(same, 5)) + float(np.percentile(diff, 95))) / 2
    log.info(
        "same-dog p5=%.3f  different-dog p95=%.3f  -> suggested min_similarity=%.3f",
        float(np.percentile(same, 5)),
        float(np.percentile(diff, 95)),
        cut,
    )
    return cut


def finetune(
    dataset: CropDataset,
    out_path: Path,
    arch: str = "mobilenet_v3_small",
    epochs: int = 20,
    batch: int = 32,
    lr: float = 1e-4,
) -> Path:
    """Metric-learning fine-tune of the embedding backbone. SKELETON.

    Only worth running when `evaluate` shows specific dogs being confused with
    each other; the zero-shot features handle visually distinct dogs already.

    Intended shape:

      1. Sample P dogs x K crops per batch (batch-hard triplet needs positives
         *within* the batch, so a plain shuffled DataLoader will not do).
      2. Forward through the backbone with its classifier stripped, exactly as
         `TorchvisionEmbedder` does, so train and inference preprocessing cannot
         drift apart.
      3. Batch-hard triplet loss on L2-normalised embeddings, or ArcFace if the
         dog count grows past a few dozen.
      4. Augment for the failure modes actually seen here -- crop jitter,
         brightness (the camera runs from dawn to dusk), horizontal flip. Do
         *not* augment hue: coat colour is the strongest identity cue there is.
      5. Early-stop on validation accuracy from `evaluate`, not on the loss.
      6. Save `{"model": state_dict, "arch": arch}` so
         `TorchvisionEmbedder(weights_path=...)` can load it directly.

    After this, re-run `enrol` -- the gallery is embedder-specific and centroids
    built with the old weights are meaningless under the new ones.
    """
    counts = {n: c for n, c in dataset.counts().items() if n in dataset.names()}
    log.info("would fine-tune %s for %d epochs on %s", arch, epochs, counts)
    raise NotImplementedError(
        "finetune() is a skeleton. Run `evaluate` first -- a pretrained backbone "
        "plus `enrol` is often enough, and this is only worth building when the "
        "confusion matrix shows two dogs actually bleeding into each other."
    )
