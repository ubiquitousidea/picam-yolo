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
    assert build_parser().parse_args(["enrol"]).embedder == "hash"


def test_shared_options_keep_their_defaults(tmp_path):
    from picam_yolo.dogid.__main__ import build_parser

    args = build_parser().parse_args(["stats"])
    assert args.root == Path("dogid")
    assert args.verbose is False
    assert build_parser().parse_args(["stats", "-v"]).verbose is True
    assert build_parser().parse_args(["-v", "stats"]).verbose is True
