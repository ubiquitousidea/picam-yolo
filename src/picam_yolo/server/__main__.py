"""Entry point for the Pi-side server: `python -m picam_yolo.server`."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading

from .cameras import (
    EXPOSURE_MODES,
    METERING_MODES,
    CameraInfo,
    PiCameraSource,
    SyntheticSource,
    build_controls,
    discover_cameras,
)
from .detector import create_detector
from .pipeline import CameraPipeline, FramePublisher

log = logging.getLogger("picam_yolo.server")


def _size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {text!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="picam-yolo-server",
        description="Capture from every attached Pi camera, run YOLO, publish over ZeroMQ.",
    )
    p.add_argument("--bind", default="tcp://0.0.0.0:5555", help="ZeroMQ PUB address")
    p.add_argument("--size", type=_size, default=(1280, 720), help="capture size, e.g. 1280x720")
    p.add_argument(
        "--cameras",
        default="all",
        help="comma-separated camera numbers to use, or 'all' (default)",
    )
    p.add_argument("--backend", choices=("ncnn", "none"), default="ncnn")
    p.add_argument("--model", default="models/yolo11n_ncnn_model", help="exported NCNN model dir")
    p.add_argument("--imgsz", type=int, default=416, help="must match the export size")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument(
        "--classes",
        default="",
        help="comma-separated COCO class ids to keep (empty = all)",
    )
    p.add_argument(
        "--detect-every",
        type=int,
        default=1,
        help="run detection on every Nth frame, reusing boxes in between",
    )
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument(
        "--threads",
        type=int,
        default=2,
        help="inference threads per camera (4 cores total on a Pi 5)",
    )
    p.add_argument(
        "--synthetic",
        type=int,
        default=0,
        metavar="N",
        help="skip hardware and publish N test-pattern streams instead",
    )
    # Exposure. Indoors, an unconstrained AE loop picks a long shutter and
    # smears anything that moves -- see the cameras.py docstring.
    e = p.add_argument_group("exposure")
    e.add_argument(
        "--exposure-mode",
        choices=EXPOSURE_MODES,
        default="normal",
        help="'short' biases AE to a fast shutter and higher gain (default: normal)",
    )
    e.add_argument(
        "--ev", type=float, default=0.0, help="exposure compensation in stops, e.g. 1.0"
    )
    e.add_argument("--metering", choices=METERING_MODES, default="centre")
    e.add_argument(
        "--exposure-us",
        type=int,
        default=None,
        help="pin the shutter in microseconds; disables AE (use with --gain)",
    )
    e.add_argument("--gain", type=float, default=None, help="analogue gain")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _select_cameras(args) -> list[CameraInfo]:
    if args.synthetic:
        return [CameraInfo(num=i, model="synthetic", camera_id=f"synth{i}") for i in range(args.synthetic)]

    cams = discover_cameras()
    if args.cameras != "all":
        wanted = {int(x) for x in args.cameras.split(",") if x.strip()}
        cams = [c for c in cams if c.num in wanted]
        missing = wanted - {c.num for c in cams}
        if missing:
            log.warning("requested camera(s) %s not detected", sorted(missing))
    return cams


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    cameras = _select_cameras(args)
    if not cameras:
        log.error(
            "no cameras detected. Check the ribbon seating and `rpicam-hello --list-cameras`, "
            "or pass --synthetic 1 to test the pipeline without hardware."
        )
        return 1
    log.info("using %d camera(s): %s", len(cameras), ", ".join(c.label for c in cameras))

    # Built once, and only when a real camera is present: build_controls imports
    # libcamera, which does not exist on a dev machine running --synthetic.
    camera_controls: dict = {}
    if any(c.model != "synthetic" for c in cameras):
        try:
            camera_controls = build_controls(
                exposure_mode=args.exposure_mode,
                ev=args.ev,
                metering=args.metering,
                exposure_us=args.exposure_us,
                gain=args.gain,
            )
        except ValueError as exc:
            log.error("%s", exc)
            return 1

    classes = [int(x) for x in args.classes.split(",") if x.strip()] or None
    try:
        publisher = FramePublisher(args.bind)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    pipelines: list[CameraPipeline] = []

    for info in cameras:
        # One detector per pipeline: Ultralytics models are not thread-safe.
        try:
            detector = create_detector(
                args.backend,
                **(
                    {}
                    if args.backend == "none"
                    else dict(
                        model_path=args.model,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        iou=args.iou,
                        classes=classes,
                        num_threads=args.threads,
                    )
                ),
            )
        except Exception:
            log.exception("could not initialise the %s detector", args.backend)
            publisher.close()
            return 1

        source = (
            SyntheticSource(info, args.size)
            if info.model == "synthetic"
            else PiCameraSource(info, args.size, controls=camera_controls)
        )
        pipelines.append(
            CameraPipeline(
                source=source,
                detector=detector,
                publisher=publisher,
                jpeg_quality=args.jpeg_quality,
                detect_every=args.detect_every,
            )
        )

    stopping = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())

    for pipe in pipelines:
        pipe.start()
    log.info("running; Ctrl-C to stop")
    stopping.wait()

    log.info("shutting down")
    for pipe in pipelines:
        pipe.stop()
    for pipe in pipelines:
        pipe.join(timeout=5.0)
    publisher.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
