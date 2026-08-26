"""Turn a dog crop into a vector.

Why an embedding rather than per-dog classes in the detector: a classifier has
to be retrained to learn a new dog, and a detection head at imgsz 416 sees a
distant dog as a handful of pixels. An embedding is trained once to make
*same dog* vectors close and *different dog* vectors far apart; after that,
enrolling a new dog is arithmetic on ~20 crops, not a training run. This is the
same reason face recognition is built this way.

Mirrors `server/detector.py` deliberately: a `Embedder` Protocol, concrete
backends, and a `create_embedder` factory keyed by name. Adding a better
backbone means adding a class and a branch, nothing else.

`HashEmbedder` is the counterpart of `NullDetector` -- it needs no torch and no
weights, so the dataset, gallery, labeller and CLI can all be exercised on a
machine with nothing installed. It is not good enough for real identification;
it exists to separate "the pipeline is broken" from "the model is bad".
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

# Every backend resizes to this before embedding. Bigger helps fine-grained ID
# but costs time; 224 is what the torchvision backbones expect.
INPUT_SIZE = 224


class Embedder(Protocol):
    """Maps a batch of BGR crops to L2-normalised row vectors."""

    dim: int
    backend: str

    def embed(self, crops: list[np.ndarray]) -> np.ndarray: ...

    def spec(self) -> dict: ...


def l2_normalise(x: np.ndarray) -> np.ndarray:
    """Row-wise unit length, so cosine similarity is a plain dot product and
    the gallery's distance threshold means the same thing for every backend."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-9)


class HashEmbedder:
    """Deterministic colour/gradient statistics. No dependencies, no weights.

    Good enough to prove the plumbing end to end; nowhere near good enough to
    tell two brown terriers apart. Use `--embedder torch` for real work.
    """

    CELLS = 2  # 2x2 spatial grid
    HUE_BINS = 16

    dim = CELLS * CELLS * HUE_BINS  # 64
    backend = "hash"

    def spec(self) -> dict:
        return {"backend": self.backend, "dim": self.dim}

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Per-cell hue histograms, concatenated.

        Deliberately *not* a mean over H, S and V: averaging the channels
        together collapses every crop onto nearly the same vector, cosine
        similarity pins at ~1.0 for everything, and the gallery rejects the
        world. Hue carries almost all of the discriminative signal at this
        fidelity, and the 2x2 grid keeps a little spatial structure so that a
        white-chested black dog differs from a uniformly black one.
        """
        import cv2

        out = np.zeros((len(crops), self.dim), dtype=np.float32)
        step = 32 // self.CELLS
        for i, crop in enumerate(crops):
            hsv = cv2.cvtColor(
                cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV
            )
            # OpenCV hue is 0..179; bin it, then weight each pixel by saturation
            # so that washed-out background contributes less than coat colour.
            bins = (hsv[..., 0].astype(np.int32) * self.HUE_BINS) // 180
            sat = hsv[..., 1].astype(np.float32) / 255.0
            feats = []
            for cy in range(self.CELLS):
                for cx in range(self.CELLS):
                    sl = (slice(cy * step, (cy + 1) * step), slice(cx * step, (cx + 1) * step))
                    feats.append(
                        np.bincount(
                            bins[sl].ravel(), weights=sat[sl].ravel(), minlength=self.HUE_BINS
                        )[: self.HUE_BINS]
                    )
            out[i] = np.concatenate(feats)
        return l2_normalise(out)


class TorchvisionEmbedder:
    """A pretrained CNN with its classifier head removed.

    Zero-shot ImageNet features already separate visually distinct dogs well
    enough to be useful, which is what makes the "label a few, enrol, done"
    loop possible before any training has happened. `train.py` can then
    fine-tune these weights with a metric loss to sharpen the boundaries
    between *similar* dogs, which is where the zero-shot features struggle.
    """

    backend = "torch"

    def spec(self) -> dict:
        return {"backend": self.backend, "dim": self.dim, "arch": self.arch,
                "weights": self.weights_path}

    def __init__(self, arch: str = "mobilenet_v3_small", weights_path: str | None = None):
        import torch
        import torchvision

        self._torch = torch
        builder = getattr(torchvision.models, arch, None)
        if builder is None:
            raise ValueError(f"unknown torchvision arch: {arch!r}")

        # weights=None when we are loading our own fine-tuned checkpoint --
        # downloading ImageNet weights only to overwrite them wastes time.
        model = builder(weights=None if weights_path else "DEFAULT")
        self._feature_dim = _strip_classifier(model)
        if weights_path:
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state.get("model", state))
            log.info("loaded fine-tuned embedder from %s", weights_path)

        model.eval()
        self._model = model
        self.dim = self._feature_dim
        self.arch = arch
        self.weights_path = weights_path

        # ImageNet normalisation, in BGR order to match this project's frames.
        self._mean = np.array([0.406, 0.456, 0.485], dtype=np.float32)
        self._std = np.array([0.225, 0.224, 0.229], dtype=np.float32)

    def _batch(self, crops: list[np.ndarray]):
        import cv2

        arr = np.stack(
            [
                (
                    cv2.resize(c, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA).astype(
                        np.float32
                    )
                    / 255.0
                    - self._mean
                )
                / self._std
                for c in crops
            ]
        )
        return self._torch.from_numpy(arr.transpose(0, 3, 1, 2))

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._torch.no_grad():
            feats = self._model(self._batch(crops))
        return l2_normalise(feats.numpy().reshape(len(crops), -1).astype(np.float32))


def _strip_classifier(model) -> int:
    """Replace the classification head with identity, returning the feature width.

    torchvision is not consistent about what the head is called, so this probes
    the two shapes that cover every backbone we are likely to try.
    """
    import torch.nn as nn

    if hasattr(model, "classifier"):
        head = model.classifier
        layers = list(head) if isinstance(head, nn.Sequential) else [head]
        width = next(l.in_features for l in layers if isinstance(l, nn.Linear))
        model.classifier = nn.Identity()
        return width
    if hasattr(model, "fc"):
        width = model.fc.in_features
        model.fc = nn.Identity()
        return width
    raise ValueError(f"cannot find a classifier head on {type(model).__name__}")


def create_embedder(backend: str = "hash", **kwargs) -> Embedder:
    """Factory keyed by backend name, mirroring `create_detector`."""
    if backend == "hash":
        return HashEmbedder()
    if backend == "torch":
        return TorchvisionEmbedder(**kwargs)
    raise ValueError(f"unknown embedder backend: {backend!r}")
