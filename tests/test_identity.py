"""Tests for live dog identification in the viewer.

Same constraints as `test_dogid.py`: the `hash` embedder and synthetic frames,
so this needs no torch, no camera and no running Pi. It covers the two pieces
that are easy to get quietly wrong -- carrying a name onto a box that has moved,
and cropping at query time the same way `capture` cropped at enrolment time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from picam_yolo.client.identity import (
    DogIdentifier,
    IdentityTracker,
    identify_frame,
    iou,
    name_color,
)
from picam_yolo.client.viewer import detection_label
from picam_yolo.dogid.dataset import CropDataset
from picam_yolo.dogid.embed import create_embedder
from picam_yolo.dogid.gallery import Gallery
from picam_yolo.protocol import Detection, FrameHeader

from test_dogid import synth_dog


def _frame_with(dogs: list[tuple[int, tuple[int, int, int, int]]], size=(720, 1280), seed=0):
    """A frame plus its header: each dog is a coloured patch pasted at its box."""
    frame = np.full((*size, 3), 40, dtype=np.uint8)
    dets = []
    for hue, (x1, y1, x2, y2) in dogs:
        patch = cv2.imdecode(
            np.frombuffer(synth_dog(hue, seed=hue + seed), np.uint8), cv2.IMREAD_COLOR
        )
        frame[y1:y2, x1:x2] = cv2.resize(patch, (x2 - x1, y2 - y1))
        dets.append(Detection(cls_id=16, name="dog", conf=0.9, box=(x1, y1, x2, y2)))
    header = FrameHeader(cam_id=0, seq=1, ts=0.0, width=size[1], height=size[0], detections=dets)
    return frame, header


# HashEmbedder bins hue into 16 buckets 11.25 wide. Synthetic dogs are jittered
# +/-4, so a hue sitting on a bucket edge splits its own centroid across two bins
# and stops matching itself -- a property of the toy embedder, not of the code
# under test. These sit mid-bucket (11.25k + 5.6) where the jitter stays put.
BIN_CENTRE_HUES = (28, 118)


def _gallery(tmp_path: Path, hues=BIN_CENTRE_HUES) -> tuple[Gallery, object]:
    """Enrol two visually distinct 'dogs', harvesting crops the way `capture` does.

    Enrolling from bare patches instead would make this fixture lie: `pad_box`
    pulls 12% of surrounding background into every real crop, which drags cosine
    similarity down by a good 0.1. Building the gallery through `CropHarvester`
    keeps enrolment and query crops the same shape -- which is the property the
    live path depends on.
    """
    from picam_yolo.dogid.capture import CropHarvester

    ds = CropDataset.open(tmp_path / "ds")
    harvester = CropHarvester(ds, novelty=0)
    for hue, name in zip(hues, ("rex", "bo")):
        before = set(ds.records)
        for i in range(6):
            x = 100 + i * 40
            frame, header = _frame_with([(hue, (x, 100, x + 200, 400))], seed=i)
            harvester.feed(frame, header)
        for cid in set(ds.records) - before:
            ds.label(cid, name)
    embedder = create_embedder("hash")
    return Gallery.build(ds, embedder, min_similarity=0.5, min_margin=0.02), embedder


# -- geometry ---------------------------------------------------------------


def test_iou_basics():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert 0.1 < iou((0, 0, 10, 10), (5, 0, 15, 10)) < 0.5


def test_name_color_is_stable_across_processes():
    """crc32, not hash() -- PYTHONHASHSEED would otherwise repaint every dog on
    each launch, which is exactly the thing the colour is there to prevent."""
    assert name_color("rex") == name_color("rex")
    assert name_color("rex") != name_color("bo")


# -- tracking ---------------------------------------------------------------


def test_tracker_carries_name_onto_a_moved_box():
    tracker = IdentityTracker()
    tracker.update([((100, 100, 300, 400), "REX")], now=0.0)
    # The dog has walked ~15% of its width since the worker last saw it.
    _, header = _frame_with([(28, (130, 110, 330, 410))])
    assert tracker.assign(header, now=0.1) == {0: "REX"}


def test_tracker_drops_a_name_that_no_longer_overlaps():
    tracker = IdentityTracker()
    tracker.update([((100, 100, 300, 400), "REX")], now=0.0)
    _, header = _frame_with([(28, (700, 100, 900, 400))])
    assert tracker.assign(header, now=0.1) == {}


def test_tracker_expires_stale_results():
    """A name must not outlive the dog that earned it, or it lands on whatever
    walks through the same patch of frame next."""
    tracker = IdentityTracker(max_age=1.0)
    tracker.update([((100, 100, 300, 400), "REX")], now=0.0)
    _, header = _frame_with([(28, (100, 100, 300, 400))])
    assert tracker.assign(header, now=0.5) == {0: "REX"}
    assert tracker.assign(header, now=5.0) == {}


def test_tracker_does_not_reuse_one_result_for_two_boxes():
    tracker = IdentityTracker()
    tracker.update([((100, 100, 300, 400), "REX")], now=0.0)
    _, header = _frame_with([(28, (100, 100, 300, 400)), (118, (110, 105, 310, 405))])
    assigned = tracker.assign(header, now=0.1)
    assert list(assigned.values()) == ["REX"]
    assert len(assigned) == 1


# -- identification ---------------------------------------------------------


def test_identify_frame_names_enrolled_dogs(tmp_path):
    gallery, embedder = _gallery(tmp_path)
    frame, header = _frame_with([(28, (100, 100, 300, 400)), (118, (600, 100, 800, 400))])
    results = identify_frame(frame, header, embedder, gallery)
    assert [m.name for _, m in results] == ["rex", "bo"]


def test_identify_frame_skips_non_dogs_and_tiny_boxes(tmp_path):
    gallery, embedder = _gallery(tmp_path)
    frame, header = _frame_with([(28, (100, 100, 300, 400))])
    header.detections.append(Detection(cls_id=0, name="person", conf=0.9, box=(0, 0, 200, 400)))
    # Below MIN_CROP_PX there is not enough detail to identify anyone.
    header.detections.append(Detection(cls_id=16, name="dog", conf=0.9, box=(0, 0, 30, 30)))
    assert len(identify_frame(frame, header, embedder, gallery)) == 1


def test_identify_frame_matches_the_crop_capture_would_have_saved(tmp_path):
    """Query-time crops must match enrolment-time crops. A drift here shows up
    as mysteriously low similarity, never as an error."""
    from picam_yolo.dogid.capture import CropHarvester

    gallery, embedder = _gallery(tmp_path)
    ds = CropDataset.open(tmp_path / "harvested")
    frame, header = _frame_with([(28, (100, 100, 300, 400))])
    CropHarvester(ds, novelty=0).feed(frame, header)

    (saved,) = list(ds.records)
    from_disk = embedder.embed([ds.image(saved)])[0]
    from_stream = identify_frame(frame, header, embedder, gallery)
    # Same pixels modulo one q92 JPEG round trip: cosine similarity ~1.
    assert float(from_disk @ embedder.embed([frame[88:412, 76:324]])[0]) > 0.99
    assert from_stream[0][1].name == "rex"


def test_unknown_dog_is_rejected_not_misnamed(tmp_path):
    gallery, embedder = _gallery(tmp_path)
    # A third, unenrolled colour, mid-bucket like the other two.
    frame, header = _frame_with([(73, (100, 100, 300, 400))])
    (_, match), = identify_frame(frame, header, embedder, gallery)
    assert match.name is None


# -- worker -----------------------------------------------------------------


def test_identifier_keeps_cameras_apart(tmp_path):
    """The IoU test compares coordinates, which say nothing about which camera
    they came from -- so results are tracked per camera."""
    gallery, embedder = _gallery(tmp_path)
    ident = DogIdentifier(gallery, embedder)
    frame0, header0 = _frame_with([(28, (100, 100, 300, 400))])
    _, header1 = _frame_with([(118, (100, 100, 300, 400))])
    header1.cam_id = 1

    ident.submit(frame0, header0)
    with ident._lock:
        jobs, ident._pending = ident._pending, {}
    for cam_id, (f, h) in jobs.items():
        ident.trackers.setdefault(cam_id, IdentityTracker()).update(
            identify_frame(f, h, embedder, gallery)
        )

    assert ident.labels(header0)[0].name == "rex"
    assert ident.labels(header1) == {}  # cam1 has never been identified


def test_identifier_submit_keeps_only_the_newest_frame(tmp_path):
    """Drop, don't queue -- a backlog here means naming a dog from a stale frame."""
    gallery, embedder = _gallery(tmp_path)
    ident = DogIdentifier(gallery, embedder)
    frame, header = _frame_with([(28, (100, 100, 300, 400))])
    for seq in range(5):
        header.seq = seq
        ident.submit(frame, header)
    assert len(ident._pending) == 1
    assert ident._pending[0][1].seq == 4


def test_identifier_ignores_frames_with_no_dogs(tmp_path):
    gallery, embedder = _gallery(tmp_path)
    ident = DogIdentifier(gallery, embedder)
    frame, header = _frame_with([])
    header.detections.append(Detection(cls_id=0, name="person", conf=0.9, box=(0, 0, 200, 400)))
    ident.submit(frame, header)
    assert ident._pending == {}


def test_identifier_thread_produces_labels(tmp_path):
    """End to end through the real worker thread."""
    import time

    gallery, embedder = _gallery(tmp_path)
    ident = DogIdentifier(gallery, embedder, min_interval=0.0)
    frame, header = _frame_with([(28, (100, 100, 300, 400))])
    ident.start()
    try:
        ident.submit(frame, header)
        for _ in range(100):
            if ident.labels(header):
                break
            time.sleep(0.02)
    finally:
        ident.stop()
    assert ident.labels(header)[0].name == "rex"


# -- drawing ----------------------------------------------------------------


def test_detection_label_variants():
    from picam_yolo.dogid.gallery import Match

    det = Detection(cls_id=16, name="dog", conf=0.87, box=(0, 0, 100, 100))
    assert detection_label(det, None)[0] == "dog 0.87"
    assert detection_label(det, Match("rex", 0.91, 0.2))[0] == "rex 0.91"
    # A rejection is shown, not hidden: the similarity that fell short is the
    # feedback that tells you whether to collect crops or lower the threshold.
    assert detection_label(det, Match(None, 0.42, 0.01))[0] == "dog ? 0.42"
    assert detection_label(det, Match("rex", 0.91, 0.2))[1] == name_color("rex")


def test_draw_overlay_runs_with_and_without_identities():
    """Smoke the drawing path itself, not just the label logic -- an out-of-range
    box or a missing identity key would otherwise only show up on a live stream."""
    from picam_yolo.client.viewer import draw_overlay
    from picam_yolo.dogid.gallery import Match

    frame, header = _frame_with([(28, (100, 100, 300, 400))])
    # A box hugging the top edge, where the label has to be pushed back inside.
    header.detections.append(Detection(cls_id=16, name="dog", conf=0.5, box=(10, 0, 200, 90)))
    before = frame.copy()

    assert draw_overlay(frame.copy(), header, 19.0, True) is not None
    out = draw_overlay(frame, header, 19.0, True, {0: Match("rex", 0.91, 0.2)})
    assert out.shape == before.shape
    assert not np.array_equal(out, before)  # something was actually drawn
