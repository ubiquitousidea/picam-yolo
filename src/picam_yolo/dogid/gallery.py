"""Match an embedding to an enrolled dog.

The gallery is the whole point of the two-stage design: adding a dog is
appending rows to a small npz, not a training run. It stores one centroid per
dog (the mean of that dog's normalised embeddings) plus the spread of its
members, and answers with a name or `None` for "not anyone I know".

**Rejecting strangers is the hard part**, and the reason `min_margin` exists
alongside `min_similarity`. Cosine similarity to the nearest centroid is high
for *any* dog once the embedder is decent, so an absolute threshold alone will
confidently call a stray Rex. Requiring the best match to beat the runner-up by
a margin catches the case where a crop sits between two enrolled dogs -- which
is exactly what an unfamiliar dog looks like in embedding space.

Negatives from the dataset (`__unknown__`, `__not_a_dog__`) are enrolled as a
rejection class for the same reason. A crop closer to the negatives than to any
dog is rejected outright, which is what turns "the detector saw a cat" into a
non-answer rather than a wrong answer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import NOT_A_DOG, UNKNOWN, CropDataset
from .embed import Embedder, l2_normalise

log = logging.getLogger(__name__)

REJECT = "__reject__"  # internal centroid name covering both negative labels
HASH_DIM = 64  # HashEmbedder.dim; identifies a legacy gallery with no recorded spec


@dataclass
class Match:
    name: str | None  # None means rejected / unknown
    confidence: float  # cosine similarity to the winning centroid
    margin: float  # how far it beat the runner-up
    runner_up: str | None = None


class Gallery:
    """Centroid-per-dog nearest-neighbour index."""

    def __init__(
        self,
        names: list[str],
        centroids: np.ndarray,
        min_similarity: float = 0.55,
        min_margin: float = 0.05,
        meta: dict | None = None,
        embedder: dict | None = None,
    ):
        if len(names) != len(centroids):
            raise ValueError("names and centroids disagree in length")
        self.names = names
        self.centroids = l2_normalise(np.asarray(centroids, dtype=np.float32))
        self.min_similarity = min_similarity
        self.min_margin = min_margin
        self.meta = meta or {}
        # Which embedder produced these centroids. A gallery is meaningless
        # under any other one, and without this recorded the mismatch surfaces
        # as a matmul shape error deep inside numpy rather than as advice.
        self.embedder = embedder or {}

    # -- build / persist ---------------------------------------------------

    @classmethod
    def build(
        cls,
        dataset: CropDataset,
        embedder: Embedder,
        split: str | None = "train",
        batch: int = 32,
        **kwargs,
    ) -> Gallery:
        """Enrol every identity in `dataset`, plus a rejection class."""
        groups: dict[str, list[str]] = {}
        for rec in dataset.identities(split=split):
            groups.setdefault(rec.label, []).append(rec.crop_id)
        for rec in dataset.negatives():
            groups.setdefault(REJECT, []).append(rec.crop_id)

        if not any(name != REJECT for name in groups):
            raise ValueError(
                "no labelled dogs in the dataset -- run `label` before `enrol`"
            )

        names, centroids, meta = [], [], {}
        for name, crop_ids in sorted(groups.items()):
            vecs = dataset.embed_ids(embedder, crop_ids, batch)
            if not len(vecs):
                raise ValueError(f"no readable crops for {name!r}")
            centroid = vecs.mean(axis=0)
            names.append(name)
            centroids.append(centroid)
            # Mean cosine similarity of members to their own centroid. A low
            # value means the crops for this dog do not agree with each other,
            # usually mislabelling or wildly varied lighting -- worth surfacing
            # rather than silently enrolling.
            unit = centroid / max(np.linalg.norm(centroid), 1e-9)
            meta[name] = {
                "n": len(crop_ids),
                "cohesion": round(float((vecs @ unit).mean()), 4),
            }
            log.info("enrolled %-16s n=%-4d cohesion=%.3f", name, len(crop_ids), meta[name]["cohesion"])

        return cls(names, np.stack(centroids), meta=meta,
                   embedder=embedder.spec() if hasattr(embedder, "spec") else {}, **kwargs)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            names=np.array(self.names),
            centroids=self.centroids,
            config=np.array(
                json.dumps(
                    {
                        "min_similarity": self.min_similarity,
                        "min_margin": self.min_margin,
                        "meta": self.meta,
                        "embedder": self.embedder,
                    }
                )
            ),
        )
        log.info("wrote gallery with %d entries to %s", len(self.names), path)

    @classmethod
    def load(cls, path: Path | str) -> Gallery:
        data = np.load(Path(path), allow_pickle=False)
        cfg = json.loads(str(data["config"]))
        return cls(
            names=[str(n) for n in data["names"]],
            centroids=data["centroids"],
            min_similarity=cfg["min_similarity"],
            min_margin=cfg["min_margin"],
            meta=cfg.get("meta", {}),
            embedder=cfg.get("embedder", {}),
        )

    # -- query -------------------------------------------------------------

    @property
    def dog_names(self) -> list[str]:
        return [n for n in self.names if n != REJECT]

    @property
    def dim(self) -> int:
        return int(self.centroids.shape[1])

    @property
    def backend(self) -> str:
        """Which embedder built this, inferring it for galleries saved before
        the spec was recorded. `HashEmbedder` is the only 64-dim backend, so the
        width identifies it; anything else came from a torch backbone."""
        recorded = self.embedder.get("backend")
        if recorded:
            return recorded
        return "hash" if self.dim == HASH_DIM else "torch"

    def check_embedder(self, embedder) -> None:
        """Refuse an embedder this gallery was not built with.

        The centroids only mean anything under the backbone that produced them,
        so a mismatch is never recoverable -- but it used to announce itself as
        `matmul: size 64 is different from 576` from inside numpy. Say what
        happened and what to type instead.
        """
        if getattr(embedder, "dim", self.dim) == self.dim:
            return
        was = self.backend
        raise SystemExit(
            f"this gallery was built with the {was!r} embedder "
            f"({self.dim}-dim) but you are using {getattr(embedder, 'backend', '?')!r} "
            f"({embedder.dim}-dim).\n"
            f"Pass --embedder {was}, or rebuild it with `enrol --embedder "
            f"{getattr(embedder, 'backend', 'torch')}`."
        )

    def nearest(self, embedding: np.ndarray) -> Match:
        """Nearest centroid with **no gate applied** -- `name` is always set.

        The gates are a *serving* decision: the viewer must not put a name on a
        stranger, so a thin margin has to become a non-answer. Labelling wants
        the opposite. A crop sitting between two centroids is precisely what a
        person is worth asking about, and `label` deliberately shows those
        first, so gating the suggestion there suppresses it on exactly the
        crops it exists for. `match` gates, this does not, and the caller picks.
        """
        vec = l2_normalise(np.asarray(embedding, dtype=np.float32).reshape(1, -1))[0]
        if vec.shape[0] != self.dim:
            raise ValueError(
                f"embedding is {vec.shape[0]}-dim but this gallery is {self.dim}-dim; "
                f"it was built with the {self.embedder.get('backend', 'unknown')!r} embedder"
            )
        sims = self.centroids @ vec
        order = np.argsort(-sims)

        best = int(order[0])
        best_name, best_sim = self.names[best], float(sims[best])
        second = int(order[1]) if len(order) > 1 else None
        runner_up = self.names[second] if second is not None else None
        margin = best_sim - float(sims[second]) if second is not None else best_sim

        return Match(best_name, best_sim, margin, runner_up=runner_up)

    def rejects(self, match: Match) -> bool:
        """Whether the serving gates turn `match` into a non-answer."""
        return (
            match.name == REJECT
            or match.confidence < self.min_similarity
            or match.margin < self.min_margin
        )

    def match(self, embedding: np.ndarray) -> Match:
        """Nearest centroid, subject to the similarity and margin gates."""
        best = self.nearest(embedding)
        if self.rejects(best):
            # `runner_up` carries what it *would* have been: the client draws it
            # as `dog ? 0.42`, and `eval` needs it to say which gate rejected.
            return Match(None, best.confidence, best.margin, runner_up=best.name)
        return best

    def match_batch(self, embeddings: np.ndarray) -> list[Match]:
        return [self.match(e) for e in np.asarray(embeddings)]
