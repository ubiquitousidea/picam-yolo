"""Round-trip test for the dog-identification pipeline.

Uses the `hash` embedder and synthetic crops, so it needs neither torch, a
camera, nor a running Pi -- the same reason `--synthetic` and `--backend none`
exist on the server side.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picam_yolo.dogid.capture import CropHarvester, dhash, hamming, pad_box
from picam_yolo.dogid.dataset import NOT_A_DOG, UNKNOWN, CropDataset
from picam_yolo.dogid.embed import create_embedder
from picam_yolo.dogid.gallery import Gallery
from picam_yolo.dogid.train import evaluate
from picam_yolo.protocol import Detection, FrameHeader


def _decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def synth_dog(hue: int, seed: int, size: int = 128) -> bytes:
    """A distinctly coloured blob, jittered so crops of one 'dog' differ."""
    rng = np.random.default_rng(seed)
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[..., 0] = (hue + rng.integers(-4, 5)) % 180
    hsv[..., 1] = np.clip(200 + rng.integers(-30, 31), 0, 255)
    hsv[..., 2] = np.clip(200 + rng.integers(-40, 41), 0, 255)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img = cv2.circle(img, (size // 2, size // 2), size // 3, (30, 30, 30), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_pad_box_clips_to_frame():
    # 12% of 40px = 4.8, expanded outward on every side.
    assert pad_box((10, 10, 50, 50), 100, 100) == (5, 5, 55, 55)
    # A box hugging the edge must not produce negative or out-of-frame bounds.
    assert pad_box((0, 0, 20, 20), 100, 100) == (0, 0, 23, 23)
    assert pad_box((80, 80, 100, 100), 100, 100) == (77, 77, 100, 100)
    # Padding must never shrink a box on any side.
    for box in [(10, 10, 50, 50), (0, 0, 20, 20), (33, 7, 91, 64)]:
        px1, py1, px2, py2 = pad_box(box, 100, 100)
        assert px1 <= box[0] and py1 <= box[1]
        assert px2 >= box[2] and py2 >= box[3]


def test_dhash_detects_near_duplicates():
    a = _decode(synth_dog(20, 1))
    assert hamming(dhash(a), dhash(a.copy())) == 0
    noisy = np.clip(a.astype(int) + 3, 0, 255).astype(np.uint8)
    assert hamming(dhash(a), dhash(noisy)) < 8


def test_dataset_is_append_only_and_deduplicates(tmp_path):
    ds = CropDataset.open(tmp_path)
    jpeg = synth_dog(20, 1)
    cid = ds.add(jpeg, "cam0", 1.0, (0, 0, 10, 10), 0.9)
    assert cid is not None
    assert ds.add(jpeg, "cam0", 2.0, (0, 0, 10, 10), 0.9) is None, "same bytes must dedupe"

    ds.label(cid, "rex")
    assert ds.records[cid].label == "rex"
    # Relabelling appends; reopening must fold the log and see the latest value.
    ds.label(cid, "bella")
    assert CropDataset.open(tmp_path).records[cid].label == "bella"


def test_dataset_survives_a_truncated_manifest(tmp_path):
    ds = CropDataset.open(tmp_path)
    ds.add(synth_dog(20, 1), "cam0", 1.0, (0, 0, 10, 10), 0.9)
    with ds.manifest.open("a") as fh:
        fh.write('{"crop_id": "hal')  # interrupted mid-write
    assert len(CropDataset.open(tmp_path).records) == 1


def test_harvester_skips_small_and_wrong_class(tmp_path):
    ds = CropDataset.open(tmp_path)
    h = CropHarvester(ds)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    header = FrameHeader(
        cam_id=0, seq=1, ts=1.0, width=640, height=480,
        detections=[
            Detection(16, "dog", 0.9, (10, 10, 30, 30)),      # too small
            Detection(15, "cat", 0.9, (100, 100, 300, 300)),  # wrong class
        ],
    )
    assert h.feed(frame, header) == 0
    assert h.skipped_small == 1


def test_gallery_round_trip_identifies_and_rejects(tmp_path):
    ds = CropDataset.open(tmp_path)
    # Two visually distinct dogs, plus a negative the gallery must reject.
    for name, hue in (("rex", 20), ("bella", 120)):
        for i in range(10):
            cid = ds.add(synth_dog(hue, i), "cam0", float(i), (0, 0, 128, 128), 0.9)
            ds.label(cid, name, split="train" if i < 8 else "val")
    for i in range(4):
        cid = ds.add(synth_dog(90, 500 + i), "cam0", float(i), (0, 0, 128, 128), 0.5)
        ds.label(cid, NOT_A_DOG)

    assert ds.names() == ["bella", "rex"]

    emb = create_embedder("hash")
    gallery = Gallery.build(ds, emb, split="train")  # default gates, not permissive ones
    assert set(gallery.dog_names) == {"rex", "bella"}

    # Persistence must not change any answer.
    path = tmp_path / "gallery.npz"
    gallery.save(path)
    reloaded = Gallery.load(path)
    probe = emb.embed([ds.image(ds.identities("val")[0].crop_id)])[0]
    assert gallery.match(probe).name == reloaded.match(probe).name

    report = evaluate(ds, emb, reloaded, split="val")
    assert report.n == 4
    assert report.accuracy >= 0.75, report.render()


def test_embeddings_separate_different_dogs(tmp_path):
    """Guards the failure that a round-trip test alone misses.

    An embedder can collapse every crop onto nearly the same vector -- cosine
    similarity pins near 1.0, argmax becomes noise, and an accuracy assertion
    can still pass by luck while the space carries no information at all. So
    assert the structure directly: same-dog pairs must be markedly closer than
    different-dog pairs.
    """
    emb = create_embedder("hash")
    rex = emb.embed([_decode(synth_dog(20, i)) for i in range(6)])
    bella = emb.embed([_decode(synth_dog(120, i)) for i in range(6)])

    within = float((rex @ rex.T).mean())
    between = float((rex @ bella.T).mean())
    assert within - between > 0.3, f"embeddings barely separate: {within:.3f} vs {between:.3f}"


def test_unknown_and_not_a_dog_are_not_enrolled_as_dogs(tmp_path):
    ds = CropDataset.open(tmp_path)
    for i in range(6):
        cid = ds.add(synth_dog(20, i), "cam0", float(i), (0, 0, 128, 128), 0.9)
        ds.label(cid, "rex")
    for i in range(3):
        cid = ds.add(synth_dog(150, 900 + i), "cam0", float(i), (0, 0, 128, 128), 0.4)
        ds.label(cid, UNKNOWN)

    gallery = Gallery.build(ds, create_embedder("hash"), split=None)
    assert gallery.dog_names == ["rex"], "reserved labels must not become identities"


# -- labeller state machine ------------------------------------------------
# The GUI loop needs a window, but the key handling is pure logic and carries
# the decisions that actually mutate the dataset, so it is worth testing.


def _labeler_with(tmp_path, n=3):
    from picam_yolo.dogid.labeler import Labeler

    ds = CropDataset.open(tmp_path)
    ids = [ds.add(synth_dog(20, i), "cam0", float(i), (0, 0, 128, 128), 0.9) for i in range(n)]
    return ds, ids, Labeler(ds)


def test_labeler_digit_assigns_known_name(tmp_path):
    ds, ids, lab = _labeler_with(tmp_path)
    ds.label(ids[0], "rex")
    lab.names = ds.names()

    rec = ds.records[ids[1]]
    assert lab.handle_key(ord("1"), rec, None) == "next"
    assert CropDataset.open(tmp_path).records[ids[1]].label == "rex"


def test_labeler_enter_accepts_suggestion_only_when_offered(tmp_path):
    ds, ids, lab = _labeler_with(tmp_path)
    rec = ds.records[ids[0]]
    # No suggestion -> Enter must not label anything.
    assert lab.handle_key(13, rec, None) == "stay"
    assert not ds.records[ids[0]].is_labelled

    assert lab.handle_key(13, rec, "bella") == "next"
    assert ds.records[ids[0]].label == "bella"


def test_labeler_typing_a_new_name(tmp_path):
    ds, ids, lab = _labeler_with(tmp_path)
    rec = ds.records[ids[0]]

    lab.handle_key(ord("n"), rec, None)
    for ch in "scout":
        lab.handle_key(ord(ch), rec, None)
    lab.handle_key(8, rec, None)  # backspace
    assert lab.handle_key(13, rec, None) == "next"
    assert ds.records[ids[0]].label == "scou"
    assert "scou" in lab.names, "a newly typed name must become a digit shortcut"


def test_labeler_escape_cancels_typing_without_labelling(tmp_path):
    ds, ids, lab = _labeler_with(tmp_path)
    rec = ds.records[ids[0]]
    lab.handle_key(ord("n"), rec, None)
    lab.handle_key(ord("z"), rec, None)
    assert lab.handle_key(27, rec, None) == "stay"
    assert not ds.records[ids[0]].is_labelled
    # esc must leave typing mode, so 'q' quits rather than typing a 'q'.
    assert lab.handle_key(ord("q"), rec, None) == "quit"


def test_labeler_splits_every_fifth_crop_to_val(tmp_path):
    ds = CropDataset.open(tmp_path)
    from picam_yolo.dogid.labeler import Labeler

    lab = Labeler(ds)
    for i in range(10):
        cid = ds.add(synth_dog(20, i), "cam0", float(i), (0, 0, 128, 128), 0.9)
        lab.handle_key(13, ds.records[cid], "rex")
    assert len(ds.identities("val")) == 2
    assert len(ds.identities("train")) == 8


# -- discarding unusable crops ---------------------------------------------


def _labelled_dataset(tmp_path):
    """Two dogs, plus one negative and one unusable crop."""
    from picam_yolo.dogid.dataset import DISCARD

    ds = CropDataset.open(tmp_path / "ds")
    ids = []
    for hue, name in ((10, "rex"), (100, "bo")):
        for i in range(4):
            cid = ds.add(synth_dog(hue, seed=hue * 10 + i), source="t", ts=0.0,
                         box=(0, 0, 128, 128), det_conf=0.9)
            ds.label(cid, name)
            ids.append(cid)
    neg = ds.add(synth_dog(60, seed=999), source="t", ts=0.0, box=(0, 0, 128, 128), det_conf=0.9)
    ds.label(neg, NOT_A_DOG)
    bad = ds.add(synth_dog(70, seed=1000), source="t", ts=0.0, box=(0, 0, 128, 128), det_conf=0.9)
    ds.label(bad, DISCARD)
    return ds, bad, neg


def test_discarded_crop_is_not_an_identity(tmp_path):
    ds, bad, _ = _labelled_dataset(tmp_path)
    assert bad not in {r.crop_id for r in ds.identities()}
    assert ds.names() == ["bo", "rex"]  # DISCARD never becomes a dog name


def test_discarded_crop_does_not_seed_the_rejection_class(tmp_path):
    """The whole point of a separate label. A crop holding two known dogs sent
    to __reject__ would teach the gallery that our own dogs look like the class
    that exists to turn them down."""
    ds, bad, neg = _labelled_dataset(tmp_path)
    negative_ids = {r.crop_id for r in ds.negatives()}
    assert neg in negative_ids
    assert bad not in negative_ids


def test_discarded_crop_is_not_offered_again(tmp_path):
    """Unlike `s` (skip), which leaves it unlabelled and back in the queue."""
    ds, bad, _ = _labelled_dataset(tmp_path)
    assert bad not in {r.crop_id for r in ds.unlabelled()}
    assert [r.crop_id for r in ds.discarded()] == [bad]


def test_gallery_excludes_discarded_crops_entirely(tmp_path):
    from picam_yolo.dogid.gallery import REJECT

    ds, _, _ = _labelled_dataset(tmp_path)
    gallery = Gallery.build(ds, create_embedder("hash"))
    assert sorted(gallery.dog_names) == ["bo", "rex"]
    # The reject centroid was built from the one true negative, not two crops.
    assert gallery.meta[REJECT]["n"] == 1


def test_labeller_m_key_discards(tmp_path):
    from picam_yolo.dogid.dataset import DISCARD
    from picam_yolo.dogid.labeler import Labeler

    ds = CropDataset.open(tmp_path / "ds")
    cid = ds.add(synth_dog(10, seed=1), source="t", ts=0.0, box=(0, 0, 128, 128), det_conf=0.9)
    rec = ds.records[cid]
    lab = Labeler(ds)
    assert lab.handle_key(ord("m"), rec, None) == "next"
    assert ds.records[cid].label == DISCARD
    assert lab.names == []  # not promoted to a dog name


# -- labelling order -------------------------------------------------------


def _conf_dataset(tmp_path):
    ds = CropDataset.open(tmp_path / "ds")
    for i, conf in enumerate((0.35, 0.95, 0.60)):
        ds.add(synth_dog(10 + i * 30, seed=i), source="t", ts=float(100 - i),
               box=(0, 0, 128, 128), det_conf=conf)
    return ds, ds.unlabelled()


def test_bootstrap_order_is_most_confident_first(tmp_path):
    """Without a gallery, the clean full-body shots must come first: they are
    what the first centroids are built from. Ordering by ascending confidence
    here would seed every dog from the detector's worst crops."""
    from picam_yolo.dogid.labeler import order_for_labelling

    ds, pending = _conf_dataset(tmp_path)
    got = [r.det_conf for r in order_for_labelling(pending, ds, None, None)]
    assert got == [0.95, 0.60, 0.35]


def test_explicit_orderings(tmp_path):
    from picam_yolo.dogid.labeler import order_for_labelling

    ds, pending = _conf_dataset(tmp_path)
    assert [r.det_conf for r in order_for_labelling(pending, ds, None, None, "confident")] == [
        0.95, 0.60, 0.35
    ]
    # Capture order, for reviewing a session as it happened.
    assert [r.ts for r in order_for_labelling(pending, ds, None, None, "captured")] == [
        98.0, 99.0, 100.0
    ]
    with pytest.raises(ValueError):
        order_for_labelling(pending, ds, None, None, "sideways")


def test_uncertain_order_uses_the_gallery_margin(tmp_path):
    """With a gallery, smallest-margin-first -- the crops that move the boundary."""
    from picam_yolo.dogid.labeler import order_for_labelling

    ds, _, _ = _labelled_dataset(tmp_path)
    embedder = create_embedder("hash")
    gallery = Gallery.build(ds, embedder)
    pending = [
        ds.records[ds.add(synth_dog(h, seed=500 + h), source="t", ts=0.0,
                          box=(0, 0, 128, 128), det_conf=0.5)]
        for h in (10, 100, 60)
    ]
    ordered = order_for_labelling(pending, ds, embedder, gallery, "uncertain")
    margins = [gallery.match(embedder.embed([ds.image(r.crop_id)])[0]).margin for r in ordered]
    assert margins == sorted(margins)


# -- CLI argument positions ------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["--embedder", "torch", "enrol"],
        ["enrol", "--embedder", "torch"],
    ],
)
def test_shared_options_work_on_either_side_of_the_subcommand(argv):
    """`--embedder torch enrol` and `enrol --embedder torch` must agree.

    argparse binds an option to the parser that declares it, so declaring these
    only on the main parser rejected them after the subcommand -- where they
    read most naturally.
    """
    from picam_yolo.dogid.__main__ import build_parser

    assert build_parser().parse_args(argv).embedder == "torch"


def test_subparser_default_does_not_clobber_a_leading_value():
    """The trap in declaring an option twice: without SUPPRESS the subparser's
    own default overwrites what the main parser already stored, so
    `--embedder torch enrol` would silently fall back to 'hash'."""
    from picam_yolo.dogid.__main__ import build_parser

    args = build_parser().parse_args(["--embedder", "torch", "enrol"])
    assert args.embedder == "torch"
    # Unset is None, not "hash": _embedder() resolves it against the gallery,
    # so "the user did not choose" has to stay distinguishable from "chose hash".
    assert build_parser().parse_args(["enrol"]).embedder is None


def test_shared_options_keep_their_defaults(tmp_path):
    from picam_yolo.dogid.__main__ import build_parser

    args = build_parser().parse_args(["stats"])
    assert args.root == Path("dogid")
    assert args.verbose is False
    assert build_parser().parse_args(["stats", "-v"]).verbose is True
    assert build_parser().parse_args(["-v", "stats"]).verbose is True



# -- embedder / gallery agreement ------------------------------------------


def _torchless_gallery(tmp_path):
    ds = CropDataset.open(tmp_path / "ds")
    for hue, name in ((10, "rex"), (100, "bo")):
        for i in range(3):
            ds.label(ds.add(synth_dog(hue, seed=hue * 7 + i), source="t", ts=0.0,
                            box=(0, 0, 128, 128), det_conf=0.9), name)
    return ds, Gallery.build(ds, create_embedder("hash"))


def test_gallery_records_the_embedder_that_built_it(tmp_path):
    _, gallery = _torchless_gallery(tmp_path)
    assert gallery.embedder["backend"] == "hash"
    assert gallery.embedder["dim"] == gallery.dim == 64


def test_embedder_spec_survives_a_save_load_round_trip(tmp_path):
    _, gallery = _torchless_gallery(tmp_path)
    gallery.save(tmp_path / "g.npz")
    assert Gallery.load(tmp_path / "g.npz").embedder["backend"] == "hash"


def test_mismatched_embedder_is_reported_not_matmul_crashed(tmp_path):
    """Regression: labelling against a 576-dim torch gallery with the default
    64-dim hash embedder used to die inside numpy with
    'size 64 is different from 576'."""
    import numpy as np

    _, gallery = _torchless_gallery(tmp_path)

    class FakeTorch:
        dim, backend = 576, "torch"

    with pytest.raises(SystemExit) as exc:
        gallery.check_embedder(FakeTorch())
    msg = str(exc.value)
    assert "--embedder hash" in msg and "576" in msg and "64" in msg

    with pytest.raises(ValueError, match="576-dim.*64-dim|64-dim.*576"):
        gallery.match(np.zeros(576, dtype=np.float32))


def test_matching_embedder_passes_the_check(tmp_path):
    _, gallery = _torchless_gallery(tmp_path)
    gallery.check_embedder(create_embedder("hash"))  # must not raise


def test_embedder_defaults_to_the_gallerys_backend(tmp_path):
    """The fix for the real failure: `label` with no --embedder must pick the
    gallery's backend, not fall back to 'hash'."""
    from picam_yolo.dogid.__main__ import _embedder, build_parser

    _, gallery = _torchless_gallery(tmp_path)
    gallery.embedder = {"backend": "hash", "dim": 64}
    args = build_parser().parse_args(["label"])
    assert _embedder(args, gallery).backend == "hash"
    # An explicit choice still wins over the gallery's.
    explicit = build_parser().parse_args(["label", "--embedder", "hash"])
    assert _embedder(explicit, gallery).backend == "hash"


# -- embedding cache -------------------------------------------------------


def test_embed_ids_caches_and_reuses(tmp_path):
    """Crops are content-addressed, so a cached vector can never go stale."""
    ds = CropDataset.open(tmp_path / "ds")
    ids = [ds.add(synth_dog(10, seed=i), source="t", ts=0.0, box=(0, 0, 128, 128), det_conf=0.9)
           for i in range(5)]

    class Counting:
        dim, backend = 64, "hash"
        calls = 0

        def spec(self):
            return {"backend": self.backend, "dim": self.dim}

        def embed(self, crops):
            type(self).calls += len(crops)
            return create_embedder("hash").embed(crops)

    emb = Counting()
    first = ds.embed_ids(emb, ids)
    assert first.shape == (5, 64) and Counting.calls == 5

    second = ds.embed_ids(emb, ids)
    assert Counting.calls == 5  # nothing recomputed
    assert np.allclose(first, second)

    # A new crop costs exactly one more embed, not a full rebuild.
    ids.append(ds.add(synth_dog(90, seed=99), source="t", ts=0.0,
                      box=(0, 0, 128, 128), det_conf=0.9))
    ds.embed_ids(emb, ids)
    assert Counting.calls == 6


def test_cache_is_keyed_by_embedder(tmp_path):
    """Two backbones must never share a cache file; their vectors are not
    comparable and mixing them would corrupt the gallery silently."""
    ds = CropDataset.open(tmp_path / "ds")

    class Fake:
        def __init__(self, backend, dim, weights=None):
            self.backend, self.dim, self._w = backend, dim, weights

        def spec(self):
            return {"backend": self.backend, "dim": self.dim, "weights": self._w}

    a = ds.embedding_cache(Fake("hash", 64))
    b = ds.embedding_cache(Fake("torch", 576))
    c = ds.embedding_cache(Fake("torch", 576, "models/finetuned.pt"))
    assert len({a, b, c}) == 3


def test_corrupt_cache_is_ignored_not_fatal(tmp_path):
    ds = CropDataset.open(tmp_path / "ds")
    cid = ds.add(synth_dog(10, seed=1), source="t", ts=0.0, box=(0, 0, 128, 128), det_conf=0.9)
    emb = create_embedder("hash")
    ds.embedding_cache(emb).write_bytes(b"not an npz")
    assert ds.embed_ids(emb, [cid]).shape == (1, 64)


def test_embed_ids_present_reports_what_it_embedded(tmp_path):
    ds = CropDataset.open(tmp_path / "ds")
    cid = ds.add(synth_dog(10, seed=1), source="t", ts=0.0, box=(0, 0, 128, 128), det_conf=0.9)
    vecs, ids = ds.embed_ids_present(create_embedder("hash"), [cid, "deadbeef" * 5])
    assert ids == [cid] and vecs.shape == (1, 64)


# -- threshold suggestion --------------------------------------------------


def _val_gallery(tmp_path):
    """Two dogs with both train and val crops, so eval/suggest have data."""
    ds = CropDataset.open(tmp_path / "ds")
    for hue, name in ((28, "rex"), (118, "bo")):
        for i in range(6):
            cid = ds.add(synth_dog(hue, seed=hue * 11 + i), source="t", ts=0.0,
                         box=(0, 0, 128, 128), det_conf=0.9)
            ds.label(cid, name, split="val" if i >= 4 else "train")
    return ds, create_embedder("hash")


def test_suggest_thresholds_returns_a_usable_pair(tmp_path):
    from picam_yolo.dogid.train import suggest_thresholds

    ds, emb = _val_gallery(tmp_path)
    gallery = Gallery.build(ds, emb)
    ms, mm = suggest_thresholds(ds, emb, gallery)
    assert 0.0 <= ms <= 1.0 and 0.0 <= mm <= 1.0


def test_suggested_thresholds_are_at_least_as_good_as_the_defaults(tmp_path):
    """The old version suggested min_similarity alone from distribution
    midpoints, and on real data proposed 0.827 where the true blocker was the
    margin gate -- a value that rejected almost everything. Whatever comes back
    must not score worse than what the gallery already had."""
    from picam_yolo.dogid.train import evaluate, suggest_thresholds

    ds, emb = _val_gallery(tmp_path)
    gallery = Gallery.build(ds, emb)
    before = evaluate(ds, emb, gallery).correct

    ms, mm = suggest_thresholds(ds, emb, gallery)
    gallery.min_similarity, gallery.min_margin = ms, mm
    assert evaluate(ds, emb, gallery).correct >= before


def test_suggest_thresholds_survives_an_empty_val_split(tmp_path):
    from picam_yolo.dogid.train import suggest_thresholds

    ds = CropDataset.open(tmp_path / "ds")
    for i in range(3):
        ds.label(ds.add(synth_dog(28, seed=i), source="t", ts=0.0,
                        box=(0, 0, 128, 128), det_conf=0.9), "rex", split="train")
    gallery = Gallery.build(ds, create_embedder("hash"))
    assert suggest_thresholds(ds, create_embedder("hash"), gallery) == (
        gallery.min_similarity, gallery.min_margin
    )


def test_eval_attributes_rejections_to_the_right_gate(tmp_path):
    """The actionable half of the report: min_margin and min_similarity have
    opposite fixes, so which one rejected has to be visible."""
    from picam_yolo.dogid.train import evaluate

    ds, emb = _val_gallery(tmp_path)
    gallery = Gallery.build(ds, emb)

    # Cosine similarity cannot exceed 1.0, so this rejects every crop on the
    # similarity gate alone -- the synthetic crops score >0.999 against their
    # own centroid, so a merely-high threshold would not fire at all.
    gallery.min_similarity, gallery.min_margin = 1.01, 0.0
    r = evaluate(ds, emb, gallery)
    assert r.rejected_similarity == r.rejected > 0
    assert r.rejected_margin == 0

    gallery.min_similarity, gallery.min_margin = 0.0, 0.999
    r = evaluate(ds, emb, gallery)
    assert r.rejected_margin == r.rejected > 0
    assert r.rejected_similarity == 0
    assert "lower --min-margin" in r.render()


# -- enrol inherits tuned gates --------------------------------------------


def test_enrol_keeps_the_previous_gallerys_gates(tmp_path, monkeypatch):
    """Regression: the gates are the tuned part of a gallery and live nowhere
    else, so re-running `enrol` after labelling more crops reset them to the
    argparse defaults -- taking a tuned min_margin of 0.01 back to 0.05, and
    accuracy from 90.9% to 0.0%, with nothing in the output to say why."""
    from picam_yolo.dogid.__main__ import build_parser, cmd_enrol

    ds = CropDataset.open(tmp_path / "ds")
    for hue, name in ((28, "rex"), (118, "bo")):
        for i in range(4):
            ds.label(ds.add(synth_dog(hue, seed=hue * 13 + i), source="t", ts=0.0,
                            box=(0, 0, 128, 128), det_conf=0.9), name)

    def run(extra):
        args = build_parser().parse_args(
            ["--root", str(tmp_path / "ds"), "enrol", *extra]
        )
        args.gallery = tmp_path / "ds" / "gallery.npz"
        assert cmd_enrol(args) == 0
        return Gallery.load(args.gallery)

    tuned = run(["--min-margin", "0.01", "--min-similarity", "0.42"])
    assert (tuned.min_margin, tuned.min_similarity) == (0.01, 0.42)

    # Re-enrol with no gates given: the tuned values must survive.
    again = run([])
    assert (again.min_margin, again.min_similarity) == (0.01, 0.42)

    # An explicit value still overrides.
    changed = run(["--min-margin", "0.03"])
    assert changed.min_margin == 0.03
    assert changed.min_similarity == 0.42  # untouched one still inherited


def test_enrol_uses_defaults_when_no_gallery_exists(tmp_path):
    from picam_yolo.dogid.__main__ import (
        DEFAULT_MIN_MARGIN,
        DEFAULT_MIN_SIMILARITY,
        build_parser,
        cmd_enrol,
    )

    ds = CropDataset.open(tmp_path / "ds")
    for i in range(3):
        ds.label(ds.add(synth_dog(28, seed=i), source="t", ts=0.0,
                        box=(0, 0, 128, 128), det_conf=0.9), "rex")
    args = build_parser().parse_args(["--root", str(tmp_path / "ds"), "enrol"])
    args.gallery = tmp_path / "ds" / "gallery.npz"
    cmd_enrol(args)
    g = Gallery.load(args.gallery)
    assert (g.min_similarity, g.min_margin) == (DEFAULT_MIN_SIMILARITY, DEFAULT_MIN_MARGIN)
