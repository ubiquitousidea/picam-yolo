"""Interactive crop labeller.

OpenCV rather than a widget toolkit, for the same reason `viewer.py` paints its
own record button: the Cocoa backend has no widgets, and adding Qt or Tk for one
window would drag a GUI stack into a project that otherwise needs none. Chrome
is painted into the frame and driven by `waitKey`.

Two things make this bearable to actually use on a few thousand crops:

**It proposes, you confirm.** Once any gallery exists, each crop arrives with a
suggested name; Enter accepts it. Labelling degenerates to a keypress per crop
for everything the model already gets right, and only disagreements cost real
attention. With no gallery yet, the first pass is manual -- enrol after ~20
crops and re-run to get suggestions for the rest.

**Not every crop is usable, and that is a third answer.** `m` discards a crop
that shows two dogs in one padded box, or a motion smear, without feeding it to
the rejection class the way `u`/`x` would. `s` skips, which leaves the crop
unlabelled and offers it again next session; `m` records the judgement once.

**Ordered so the label budget buys the most.** Once a gallery exists, crops come
least-confident first, so the ones that would most improve the boundary are seen
first -- labelling 200 well-chosen crops beats labelling 2000 arbitrary ones,
which matters because the budget here is a person's evening. Before a gallery
exists the order inverts: the *most* confident crops are the clean, full-body
shots that make good centroids. See `order_for_labelling`.

Text entry is hand-rolled: `waitKey` gives one keycode at a time, so a new dog's
name is accumulated into a buffer and drawn into the window.
"""

from __future__ import annotations

import logging

import numpy as np

from .dataset import DISCARD, NOT_A_DOG, UNKNOWN, CropDataset, CropRecord
from .embed import Embedder
from .gallery import Gallery

log = logging.getLogger(__name__)

_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX, avoided at import so cv2 stays lazy
PANEL_H = 150
CANVAS_W, CANVAS_H = 720, 540
VAL_EVERY = 5  # every Nth labelled crop goes to validation


ORDERINGS = ("auto", "uncertain", "confident", "captured")


def order_for_labelling(
    records: list[CropRecord],
    dataset: CropDataset,
    embedder: Embedder | None,
    gallery: Gallery | None,
    order: str = "auto",
) -> list[CropRecord]:
    """Decide which crops to put in front of the labeller first.

    The right order is not the same before and after a gallery exists, and
    getting this backwards is expensive:

    - **Bootstrapping** (no gallery). Show the detector's *most* confident crops.
      Those are the full-body, well-framed, single-dog shots that make clean
      centroids. Measured on the first real session, the 40 least-confident of
      1505 crops ran 0.35-0.37 against a median of 0.75, and were mostly dark,
      part-occluded, or held two overlapping dogs. Seeding a dog from those
      poisons every suggestion that follows, because everything downstream is
      built on the centroid they produce.
    - **Refining** (a gallery exists). Show the crops the gallery is least sure
      of -- smallest margin between the top two centroids. Those are the ones
      that actually move the boundary, which is what makes labelling 200
      well-chosen crops beat labelling 2000 arbitrary ones.

    `auto` picks between the two on whether a gallery was supplied, which is
    almost always what you want; the rest are explicit overrides.
    """
    if order not in ORDERINGS:
        raise ValueError(f"order must be one of {ORDERINGS}")
    if order == "captured":
        return sorted(records, key=lambda r: r.ts)
    if order == "confident" or (order == "auto" and (embedder is None or gallery is None)):
        return sorted(records, key=lambda r: -r.det_conf)
    if embedder is None or gallery is None:
        # "uncertain" asked for explicitly, but nothing to be uncertain against.
        log.warning("no gallery, so ordering by detector confidence instead")
        return sorted(records, key=lambda r: -r.det_conf)

    scored = []
    for rec in records:
        try:
            vec = embedder.embed([dataset.image(rec.crop_id)])[0]
        except (FileNotFoundError, ValueError):
            continue
        scored.append((gallery.match(vec).margin, rec))
    return [rec for _, rec in sorted(scored, key=lambda pair: pair[0])]


def letterbox(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit `img` into a fixed canvas without distorting it -- aspect ratio is a
    cue for dog shape, so stretching the preview would mislead the labeller."""
    import cv2

    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = (height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


class Labeler:
    """Walks unlabelled crops and writes decisions back to the dataset."""

    def __init__(
        self,
        dataset: CropDataset,
        embedder: Embedder | None = None,
        gallery: Gallery | None = None,
    ):
        self.dataset = dataset
        self.embedder = embedder
        self.gallery = gallery
        self.names: list[str] = dataset.names()
        self._typing: str | None = None  # None = not entering a name
        self.labelled = 0

    # -- suggestion --------------------------------------------------------

    def suggest(self, rec: CropRecord) -> tuple[str | None, float]:
        if self.embedder is None or self.gallery is None:
            return None, 0.0
        try:
            vec = self.embedder.embed([self.dataset.image(rec.crop_id)])[0]
        except (FileNotFoundError, ValueError):
            return None, 0.0
        m = self.gallery.match(vec)
        return m.name, m.confidence

    # -- drawing -----------------------------------------------------------

    def _panel(self, rec: CropRecord, idx: int, total: int, suggestion, conf) -> np.ndarray:
        import cv2

        panel = np.full((PANEL_H, CANVAS_W, 3), 32, dtype=np.uint8)

        def put(text, x, y, scale=0.5, color=(230, 230, 230), thick=1):
            cv2.putText(panel, text, (x, y), _FONT, scale, color, thick, cv2.LINE_AA)

        if self._typing is not None:
            put("new dog name:", 12, 30, 0.6, (120, 220, 255))
            put(self._typing + "_", 12, 62, 0.8, (255, 255, 255), 2)
            put("enter = save    esc = cancel", 12, 95)
            return panel

        put(f"[{idx + 1}/{total}]  conf {rec.det_conf:.2f}  {rec.crop_id[:10]}", 12, 24)
        if suggestion:
            put(f"suggestion: {suggestion}  ({conf:.2f})", 12, 52, 0.7, (120, 255, 160), 2)
            put("enter = accept", 400, 52, 0.5, (120, 255, 160))
        else:
            put("no suggestion (enrol a gallery to get them)", 12, 52, 0.55, (160, 160, 160))

        for i, name in enumerate(self.names[:9]):
            put(f"{i + 1}:{name}", 12 + (i % 5) * 140, 82 + (i // 5) * 22, 0.5, (200, 220, 255))

        put(
            "n new   u unknown   x not-a-dog   m discard   s skip   b back   q quit",
            12, PANEL_H - 12, 0.46, (170, 170, 170),
        )
        return panel

    def render(self, rec: CropRecord, idx: int, total: int, suggestion, conf) -> np.ndarray:
        img = self.dataset.image(rec.crop_id)
        return np.vstack([letterbox(img, CANVAS_W, CANVAS_H), self._panel(rec, idx, total, suggestion, conf)])

    # -- input -------------------------------------------------------------

    def _commit(self, rec: CropRecord, label: str) -> None:
        # Deterministic split by count rather than random, so a re-run of the
        # same labelling session produces the same train/val partition and the
        # eval number stays comparable.
        split = "val" if self.labelled % VAL_EVERY == VAL_EVERY - 1 else "train"
        self.dataset.label(rec.crop_id, label, split=split)
        self.labelled += 1
        if label not in self.names and label not in (UNKNOWN, NOT_A_DOG, DISCARD):
            self.names = self.dataset.names()

    def handle_key(self, key: int, rec: CropRecord, suggestion: str | None) -> str:
        """Returns one of: 'next', 'back', 'stay', 'quit'."""
        if self._typing is not None:
            if key in (13, 10):  # enter
                name = self._typing.strip()
                self._typing = None
                if name:
                    self._commit(rec, name)
                    return "next"
                return "stay"
            if key == 27:  # esc
                self._typing = None
                return "stay"
            if key in (8, 127):  # backspace
                self._typing = self._typing[:-1]
                return "stay"
            if 32 <= key < 127:
                self._typing += chr(key)
            return "stay"

        if key in (ord("q"), 27):
            return "quit"
        if key == ord("s"):
            return "next"
        if key == ord("b"):
            return "back"
        if key == ord("n"):
            self._typing = ""
            return "stay"
        if key == ord("u"):
            self._commit(rec, UNKNOWN)
            return "next"
        if key == ord("x"):
            self._commit(rec, NOT_A_DOG)
            return "next"
        if key == ord("m"):
            # Unusable, not negative: two dogs in one box, a blurred smear, a
            # dog half out of frame. Recorded so it is never offered again,
            # and excluded from the rejection class -- see the dataset module
            # docstring for why those must not be the same thing.
            self._commit(rec, DISCARD)
            return "next"
        if key in (13, 10) and suggestion:
            self._commit(rec, suggestion)
            return "next"
        if ord("1") <= key <= ord("9"):
            i = key - ord("1")
            if i < len(self.names):
                self._commit(rec, self.names[i])
                return "next"
        return "stay"

    # -- loop --------------------------------------------------------------

    def run(self, records: list[CropRecord], window: str = "label dogs") -> int:
        import cv2

        if not records:
            log.info("nothing left to label")
            return 0

        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        idx = 0
        suggestion, conf = self.suggest(records[0])
        try:
            while 0 <= idx < len(records):
                rec = records[idx]
                cv2.imshow(window, self.render(rec, idx, len(records), suggestion, conf))
                key = cv2.waitKey(20) & 0xFF
                if key == 255:
                    continue
                action = self.handle_key(key, rec, suggestion)
                if action == "quit":
                    break
                if action in ("next", "back"):
                    idx += 1 if action == "next" else -1
                    idx = max(0, idx)
                    if idx < len(records):
                        suggestion, conf = self.suggest(records[idx])
        finally:
            cv2.destroyWindow(window)

        log.info("labelled %d crop(s)", self.labelled)
        return self.labelled
