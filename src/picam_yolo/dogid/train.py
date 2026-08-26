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
    # Which gate did the rejecting. Reported separately because they have
    # opposite fixes, and guessing wrong is expensive: on the first real
    # gallery every rejection was the margin gate while min_similarity turned
    # nobody away, so tuning min_similarity could only have made it worse.
    rejected_similarity: int = 0
    rejected_margin: int = 0
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
        ]
        if self.rejected:
            lines.append(
                f"  rejections: {self.rejected_margin} by min_margin, "
                f"{self.rejected_similarity} by min_similarity"
            )
            if self.rejected_margin > self.rejected_similarity:
                lines.append(
                    "  -> the margin gate is what is turning them away; lower "
                    "--min-margin, not --min-similarity"
                )
        lines += ["", "confusion (row = truth, col = predicted):"]
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
        match = gallery.match(vec)
        predicted = match.name
        report.n += 1
        key = predicted or "(rejected)"
        report.confusion.setdefault(rec.label, {})
        report.confusion[rec.label][key] = report.confusion[rec.label].get(key, 0) + 1
        if predicted == rec.label:
            report.correct += 1
        elif predicted is None:
            report.rejected += 1
            # Attribute the rejection. Similarity is checked first, matching
            # the order of the gates in Gallery.match.
            if match.confidence < gallery.min_similarity:
                report.rejected_similarity += 1
            else:
                report.rejected_margin += 1
    return report


def suggest_thresholds(
    dataset: CropDataset, embedder: Embedder, gallery: Gallery
) -> tuple[float, float]:
    """Pick `min_similarity` and `min_margin` from the data, by what actually
    scores best on the validation split.

    The earlier version suggested `min_similarity` alone, from the midpoint of
    the same-dog and different-dog similarity distributions. That is wrong
    whenever those distributions overlap -- and with two dogs of similar build
    in the same room they overlap badly. On the first real gallery it read
    same-dog p5=0.770 against different-dog p95=0.885 and proposed 0.827, which
    would have rejected almost every crop; meanwhile `min_similarity` was
    turning nobody away and the *margin* gate was rejecting 8 of 11.

    So: sweep both gates and return the pair that maximises validation accuracy,
    breaking ties toward the stricter setting, since a stricter gate rejects a
    stranger where a looser one names them.
    """
    val = {r.crop_id: r for r in dataset.identities(split="val")}
    if not val:
        log.warning("no validation crops -- label more, every %dth goes to val", 5)
        return gallery.min_similarity, gallery.min_margin

    vecs, ids = dataset.embed_ids_present(embedder, list(val))
    if not len(vecs):
        return gallery.min_similarity, gallery.min_margin

    # Score once, then sweep the gates arithmetically rather than re-matching.
    scored = []
    for cid, vec in zip(ids, vecs):
        unit = vec / max(np.linalg.norm(vec), 1e-9)
        sims = gallery.centroids @ unit
        order = np.argsort(-sims)
        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else None
        margin = float(sims[best]) - (float(sims[second]) if second is not None else 0.0)
        scored.append((gallery.names[best], float(sims[best]), margin, val[cid].label))

    sims = sorted({round(s, 3) for _, s, _, _ in scored})
    margins = sorted({round(m, 3) for _, _, m, _ in scored})
    candidates_s = [0.0] + [round(v - 0.001, 3) for v in sims]
    candidates_m = [0.0] + [round(v - 0.001, 3) for v in margins]

    # Lexicographic: most correct wins; ties go to the stricter pair, because a
    # stricter gate rejects a stranger where a looser one confidently names them.
    best_key, best_pair = (-1, -1.0), (gallery.min_similarity, gallery.min_margin)
    for ms in candidates_s:
        for mm in candidates_m:
            hits = sum(
                1
                for name, sim, margin, truth in scored
                if sim >= ms and margin >= mm and name == truth
            )
            key = (hits, ms + mm)
            if key > best_key:
                best_key, best_pair = key, (ms, mm)
    best_score = best_key[0]

    log.info(
        "best on %d val crop(s): min_similarity=%.3f min_margin=%.3f -> %d/%d correct",
        len(scored), best_pair[0], best_pair[1], best_score, len(scored),
    )
    if len(scored) < 30:
        log.warning(
            "only %d validation crops -- treat these as a hint, not a setting; "
            "label more before trusting them", len(scored)
        )
    log.info("apply with: enrol --min-similarity %.3f --min-margin %.3f", *best_pair)
    return best_pair


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
