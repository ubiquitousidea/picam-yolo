#!/usr/bin/env python3
"""Compare detector candidates on real frames from this camera.

Motivation: our dogs are sometimes detected as `cat` or `teddy bear`. That is
not merely a wrong caption -- `client/identity.py` and `dogid/capture.py` filter
detections to the literal name "dog", so a misread dog silently drops out of both
identification and crop harvesting. Before paying for a bigger model on a Pi that
already spends most of its frame budget on inference, measure whether a bigger
model (or a bigger `--imgsz`) actually fixes it.

Two subcommands:

    grab   subscribe to the Pi and archive raw JPEG payloads + their headers
    run    score one or more (weights, imgsz) candidates over that archive

Why `grab` rather than reusing `recordings/`: the viewer records *annotated*
frames by default (`Viewer.record_raw` is False, so `recorder.write` runs after
`draw_overlay`), and re-detecting on pixels with boxes and HUD text painted into
them measures the wrong thing. `dogid/crops/` is clean but already cropped, which
removes exactly the small-object regime that causes the confusion. So `grab`
archives the JPEG bytes verbatim off the wire -- no decode, no re-encode, the
exact pixels the client would see.

Two caveats on the numbers this prints:

* **The timings are useless for the Pi.** They are torch on this machine, not
  NCNN on two pinned Cortex-A76 cores. They rank candidates by relative cost;
  the real figure has to come from the Pi's own `infer` log line.
* **The accuracy comparison is valid.** The NCNN export is a graph conversion,
  not a quantisation, so class behaviour tracks the `.pt` closely enough to
  choose between candidates.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

log = logging.getLogger("bench_model")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# The classes a dog actually gets confused for. `grab --watch` and `run` both
# score against this set: counting every COCO class would drown the signal in
# sofas and people.
CONFUSABLE = ("dog", "cat", "teddy bear", "bear", "sheep", "horse", "cow")


# --- grab -------------------------------------------------------------------


def cmd_grab(args: argparse.Namespace) -> int:
    """Archive raw frames from the live stream into a corpus directory."""
    import zmq

    # `--host rpi` is usually an ssh-config alias with no DNS record, so connect
    # through the same resolver the viewer and `dogid capture` use. Without it
    # zmq connects lazily to an unresolvable name and simply receives nothing.
    from picam_yolo.client.__main__ import resolve_host
    from picam_yolo.protocol import FrameHeader, topic_for

    watch = {c.strip() for c in args.watch.split(",") if c.strip()}
    corpus = Path(args.corpus)
    (corpus / "frames").mkdir(parents=True, exist_ok=True)
    manifest = (corpus / "manifest.jsonl").open("a")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    host = resolve_host(args.host)
    log.info("subscribing to tcp://%s:%d", host, args.port)
    sock.connect(f"tcp://{host}:{args.port}")
    if args.cameras:
        for cam in args.cameras:
            sock.setsockopt(zmq.SUBSCRIBE, topic_for(cam))
    else:
        sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVTIMEO, 5000)

    deadline = time.time() + args.seconds
    seen = kept = 0
    labels: Counter[str] = Counter()
    try:
        while time.time() < deadline and kept < args.max_frames:
            try:
                _topic, raw, jpeg = sock.recv_multipart()
            except zmq.Again:
                log.warning("no frames for 5s -- is the server running?")
                continue
            header = FrameHeader.from_bytes(raw)
            seen += 1

            hits = [d for d in header.detections if not watch or d.name in watch]
            if watch and not hits:
                continue
            # One frame per `--every` qualifying frames. Consecutive frames of a
            # dozing dog are near-identical and would let one pose dominate the
            # comparison.
            if seen % args.every:
                continue

            name = f"cam{header.cam_id}_{header.seq:08d}.jpg"
            (corpus / "frames" / name).write_bytes(jpeg)
            manifest.write(
                json.dumps(
                    {
                        "file": name,
                        "cam_id": header.cam_id,
                        "seq": header.seq,
                        "ts": header.ts,
                        "width": header.width,
                        "height": header.height,
                        # What the *live* model called it, so `run` can report
                        # the comparison as "frames nano called cat".
                        "baseline": [
                            {"name": d.name, "conf": round(d.conf, 4), "box": list(d.box)}
                            for d in header.detections
                        ],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            # Flush per record. This runs for an hour waiting for a dog to walk
            # past, and buffered writes mean an interrupted session leaves the
            # frames on disk with an empty manifest -- the baseline labels, which
            # cannot be recovered afterwards, would be the part that is lost.
            manifest.flush()
            kept += 1
            labels.update(d.name for d in hits)
    finally:
        sock.close(linger=0)
        manifest.close()

    log.info("saw %d frames, archived %d to %s", seen, kept, corpus)
    if labels:
        log.info("baseline labels in archived frames: %s", dict(labels.most_common()))
    if not kept:
        log.warning("archived nothing -- was anything in %s on camera?", sorted(watch))
    return 0


# --- run --------------------------------------------------------------------


def load_corpus(corpus: Path) -> list[dict]:
    manifest = corpus / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"no manifest at {manifest} -- run `bench_model.py grab` first")
    records = []
    for line in manifest.read_text().splitlines():
        if line.strip():
            # Later records for a file win, matching CropDataset's fold.
            records.append(json.loads(line))
    by_file = {r["file"]: r for r in records}
    return list(by_file.values())


def top_label(boxes, interest: set[str]) -> str | None:
    """The highest-confidence box among the classes we care about.

    Per *frame*, not per box. Raw box counts mix in genuine multi-subject frames
    -- two dogs, or a dog beside a real teddy bear -- and would move for reasons
    that have nothing to do with the confusion being measured.
    """
    hits = [b for b in boxes if b["name"] in interest]
    if not hits:
        return None
    return max(hits, key=lambda b: b["conf"])["name"]


def score_candidate(weights: str, imgsz: int, records, corpus: Path, args) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights)
    interest = set(args.interest.split(","))

    frames: Counter[str] = Counter()
    raw: Counter[str] = Counter()
    dog_confs: list[float] = []
    elapsed = 0.0

    for rec in records:
        path = corpus / "frames" / rec["file"]
        t0 = time.perf_counter()
        results = model.predict(
            str(path), imgsz=imgsz, conf=args.conf, iou=args.iou, verbose=False
        )
        elapsed += time.perf_counter() - t0
        boxes = []
        if results:
            r = results[0]
            for b in r.boxes:
                cls_id = int(b.cls.item())
                boxes.append({"name": r.names.get(cls_id, str(cls_id)), "conf": float(b.conf.item())})
        raw.update(b["name"] for b in boxes if b["name"] in interest)
        dog_confs.extend(b["conf"] for b in boxes if b["name"] == "dog")
        frames[top_label(boxes, interest) or "(none)"] += 1

    n = max(len(records), 1)
    return {
        "candidate": f"{Path(weights).stem}@{imgsz}",
        "frames": frames,
        "raw": raw,
        "dog_rate": 100.0 * frames.get("dog", 0) / n,
        "mean_dog_conf": sum(dog_confs) / len(dog_confs) if dog_confs else 0.0,
        "ms": 1000.0 * elapsed / n,
    }


def cmd_run(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus)
    records = load_corpus(corpus)
    if not records:
        raise SystemExit(f"{corpus} is empty")
    interest = set(args.interest.split(","))
    log.info("scoring %d frames from %s", len(records), corpus)

    # The live model's own verdict, as the row every candidate is compared to.
    base: Counter[str] = Counter()
    for rec in records:
        base[top_label(rec.get("baseline", []), interest) or "(none)"] += 1

    rows = []
    for spec in args.candidate:
        weights, _, size = spec.partition(":")
        rows.append(score_candidate(weights, int(size or 416), records, corpus, args))

    order = sorted(interest, key=lambda c: -(base.get(c, 0)))
    cols = [c for c in order if base.get(c) or any(r["frames"].get(c) for r in rows)]

    n = len(records)
    head = f"{'candidate':<22}" + "".join(f"{c:>13}" for c in cols)
    head += f"{'(none)':>9}{'dog%':>8}{'conf':>7}{'ms/frame':>10}"
    print(f"\nper-frame top label, {n} frames, conf>={args.conf}\n")
    print(head)
    print("-" * len(head))
    print(
        f"{'live (as captured)':<22}"
        + "".join(f"{base.get(c, 0):>13}" for c in cols)
        + f"{base.get('(none)', 0):>9}{100.0 * base.get('dog', 0) / n:>7.1f}%{'':>7}{'':>10}"
    )
    for r in rows:
        print(
            f"{r['candidate']:<22}"
            + "".join(f"{r['frames'].get(c, 0):>13}" for c in cols)
            + f"{r['frames'].get('(none)', 0):>9}"
            f"{r['dog_rate']:>7.1f}%{r['mean_dog_conf']:>7.2f}{r['ms']:>10.1f}"
        )

    print("\nraw box counts (secondary -- a frame can hold several):")
    for r in rows:
        print(f"  {r['candidate']:<20} {dict(r['raw'].most_common())}")
    print(
        "\nms/frame is torch on this machine and says nothing about the Pi's NCNN\n"
        "inference; take that from the server's own `infer` log line."
    )
    return 0


# --- cli --------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grab", help="archive raw frames from the live stream")
    g.add_argument("--host", default="rpi")
    g.add_argument("--port", type=int, default=5555)
    g.add_argument("--corpus", default="bench", help="output directory")
    g.add_argument("--seconds", type=float, default=300.0)
    g.add_argument("--max-frames", type=int, default=400)
    g.add_argument(
        "--watch",
        default=",".join(CONFUSABLE),
        help="only archive frames where the live model reported one of these "
        "(empty = archive everything)",
    )
    g.add_argument(
        "--every",
        type=int,
        default=4,
        help="keep 1 in N qualifying frames; consecutive frames are near-identical",
    )
    g.add_argument("--cameras", type=int, nargs="*", default=None)
    g.set_defaults(func=cmd_grab)

    r = sub.add_parser("run", help="score candidates over an archived corpus")
    r.add_argument("--corpus", default="bench")
    r.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="WEIGHTS:IMGSZ",
        help="e.g. yolo11s.pt:416; repeat for each candidate",
    )
    r.add_argument("--conf", type=float, default=0.35, help="matches the server default")
    r.add_argument("--iou", type=float, default=0.45, help="matches the server default")
    r.add_argument("--interest", default=",".join(CONFUSABLE))
    r.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
