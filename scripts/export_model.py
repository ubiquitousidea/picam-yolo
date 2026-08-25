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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="yolo11n.pt", help="checkpoint, downloaded if absent")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--outdir", type=Path, default=Path("models"))
    args = p.parse_args()

    from ultralytics import YOLO

    print(f"exporting {args.weights} at imgsz={args.imgsz} ...")
    exported = Path(YOLO(args.weights).export(format="ncnn", imgsz=args.imgsz))

    args.outdir.mkdir(parents=True, exist_ok=True)
    target = args.outdir / exported.name
    if target.resolve() != exported.resolve():
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(exported), str(target))

    print(f"\nwrote {target}")
    print(f"run the server with: --model {target} --imgsz {args.imgsz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
