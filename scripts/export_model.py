#!/usr/bin/env python3
"""Export a YOLO checkpoint to the NCNN format the server loads.

NCNN bakes the input resolution into the exported graph, so the --imgsz used
here must match the server's --imgsz exactly. Run this once per (model, imgsz)
pair. It needs torch, so it is usually faster to run on a laptop and rsync the
result over than to export on the Pi -- but either works.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _warn_if_different_imgsz(target: Path, imgsz: int) -> None:
    """Say so out loud when this export replaces one built at another size.

    NCNN bakes the input dimensions into the graph and the server cannot detect a
    mismatch -- it produces silently wrong boxes rather than an error. Since the
    directory is named after the weights alone, replacing a 416 export with a 640
    one leaves every `--model models/yolo11n_ncnn_model --imgsz 416` invocation in
    the repo quietly broken. Use --outname to keep both.
    """
    meta = target / "metadata.yaml"
    if not meta.exists():
        return
    for line in meta.read_text().splitlines():
        if line.startswith("imgsz:") and str(imgsz) not in line:
            print(
                f"WARNING: {target} was exported at {line.split(':', 1)[1].strip()}, "
                f"replacing it with imgsz={imgsz}. Anything still passing the old "
                "--imgsz will get silently wrong boxes. Use --outname to keep both."
            )
            return


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="yolo11n.pt", help="checkpoint, downloaded if absent")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--outdir", type=Path, default=Path("models"))
    p.add_argument(
        "--outname",
        default=None,
        help="output directory name (default: <weights-stem>_ncnn_model). Ultralytics "
        "names the export after the weights alone, so two exports of the same "
        "checkpoint at different --imgsz collide; name them apart with this.",
    )
    args = p.parse_args()

    from ultralytics import YOLO

    print(f"exporting {args.weights} at imgsz={args.imgsz} ...")
    exported = Path(YOLO(args.weights).export(format="ncnn", imgsz=args.imgsz))

    args.outdir.mkdir(parents=True, exist_ok=True)
    target = args.outdir / (args.outname or exported.name)
    if target.resolve() != exported.resolve():
        if target.exists():
            _warn_if_different_imgsz(target, args.imgsz)
            shutil.rmtree(target)
        shutil.move(str(exported), str(target))

    print(f"\nwrote {target}")
    print(f"run the server with: --model {target} --imgsz {args.imgsz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
