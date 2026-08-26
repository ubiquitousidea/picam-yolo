"""Entry point for the dog-identification workflow: `python -m picam_yolo.dogid`.

The loop this drives:

    capture -> label -> enrol -> eval -> (finetune) -> back to capture

`capture` costs the Pi nothing: it subscribes to the stream the server already
publishes and crops the boxes YOLO already found. Everything else is local.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_ROOT = Path("dogid")


def _dataset(args):
    from .dataset import CropDataset

    return CropDataset.open(args.root)


def _embedder(args):
    from .embed import create_embedder

    kwargs = {}
    if args.embedder == "torch":
        kwargs = {"arch": args.arch, "weights_path": args.weights}
    return create_embedder(args.embedder, **kwargs)


def _gallery(args, required: bool = True):
    from .gallery import Gallery

    path = Path(args.gallery)
    if not path.exists():
        if required:
            raise SystemExit(f"no gallery at {path} -- run `enrol` first")
        return None
    return Gallery.load(path)


def cmd_capture(args) -> int:
    from ..client.__main__ import resolve_host
    from .capture import harvest_stream

    ds = _dataset(args)
    kept = harvest_stream(
        ds,
        host=resolve_host(args.host),
        port=args.port,
        seconds=args.seconds,
        cameras=args.cameras,
        min_conf=args.min_conf,
        novelty=args.novelty,
    )
    print(f"kept {kept} new crop(s); dataset now holds {len(ds.records)}")
    return 0


def cmd_label(args) -> int:
    from .labeler import Labeler, order_for_labelling

    ds = _dataset(args)
    pending = ds.unlabelled()
    if not pending:
        print("nothing unlabelled -- run `capture` first")
        return 0

    gallery = _gallery(args, required=False)
    embedder = _embedder(args) if gallery else None
    if gallery:
        print(f"suggesting from {len(gallery.dog_names)} enrolled dog(s)")
    ordered = order_for_labelling(pending, ds, embedder, gallery, args.order)[: args.limit]

    n = Labeler(ds, embedder, gallery).run(ordered)
    print(f"labelled {n}; {len(ds.unlabelled())} still unlabelled")
    return 0


def cmd_enrol(args) -> int:
    from .gallery import Gallery

    ds = _dataset(args)
    gallery = Gallery.build(
        ds,
        _embedder(args),
        min_similarity=args.min_similarity,
        min_margin=args.min_margin,
    )
    gallery.save(args.gallery)
    print(f"enrolled {len(gallery.dog_names)} dog(s): {', '.join(gallery.dog_names)}")
    return 0


def cmd_eval(args) -> int:
    from .train import evaluate, suggest_threshold

    ds, emb, gal = _dataset(args), _embedder(args), _gallery(args)
    report = evaluate(ds, emb, gal)
    if not report.n:
        print("no validation crops yet -- label more (every 5th goes to val)")
        return 1
    print(report.render())
    if args.suggest_threshold:
        suggest_threshold(ds, emb, gal)
    return 0


def cmd_finetune(args) -> int:
    from .train import finetune

    finetune(_dataset(args), Path(args.out), arch=args.arch, epochs=args.epochs)
    return 0


def cmd_stats(args) -> int:
    ds = _dataset(args)
    counts = ds.counts()
    if not counts:
        print(f"{args.root}: empty -- run `capture`")
        return 0
    width = max(len(k) for k in counts)
    print(f"{args.root}: {len(ds.records)} crop(s)")
    for name, n in counts.items():
        print(f"  {name:<{width}}  {n}")
    train_n, val_n = len(ds.identities("train")), len(ds.identities("val"))
    print(f"\nidentity crops: {train_n} train / {val_n} val")
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Local, like every other import here: the CLI must build without cv2.
    from .labeler import ORDERINGS

    p = argparse.ArgumentParser(prog="picam_yolo.dogid", description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="dataset directory")
    p.add_argument("--gallery", type=Path, default=None, help="default: <root>/gallery.npz")
    p.add_argument("--embedder", choices=("hash", "torch"), default="hash",
                   help="'hash' needs no torch and is for smoke-testing only")
    p.add_argument("--arch", default="mobilenet_v3_small")
    p.add_argument("--weights", default=None, help="fine-tuned embedder checkpoint")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="harvest dog crops from the live stream")
    c.add_argument("--host", default="rpi")
    c.add_argument("--port", type=int, default=5555)
    c.add_argument("--seconds", type=float, default=60)
    c.add_argument("--cameras", type=int, nargs="*", default=None)
    c.add_argument("--min-conf", type=float, default=0.4)
    c.add_argument("--novelty", type=int, default=8, help="0 keeps near-duplicates")
    c.set_defaults(func=cmd_capture)

    c = sub.add_parser("label", help="assign identities to crops")
    c.add_argument("--limit", type=int, default=500)
    c.add_argument(
        "--order",
        choices=ORDERINGS,
        default="auto",
        help="auto: most-confident first while bootstrapping, least-certain once "
             "a gallery exists (default)",
    )
    c.set_defaults(func=cmd_label)

    c = sub.add_parser("enrol", help="build the gallery from labelled crops")
    c.add_argument("--min-similarity", type=float, default=0.55)
    c.add_argument("--min-margin", type=float, default=0.05)
    c.set_defaults(func=cmd_enrol)

    c = sub.add_parser("eval", help="score the gallery on held-out crops")
    c.add_argument("--suggest-threshold", action="store_true")
    c.set_defaults(func=cmd_eval)

    c = sub.add_parser("finetune", help="sharpen the embedder (skeleton)")
    c.add_argument("--out", default="models/dog_embedder.pt")
    c.add_argument("--epochs", type=int, default=20)
    c.set_defaults(func=cmd_finetune)

    sub.add_parser("stats", help="what is in the dataset").set_defaults(func=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if args.gallery is None:
        args.gallery = Path(args.root) / "gallery.npz"
    try:
        return args.func(args)
    except KeyboardInterrupt:
        log.info("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
