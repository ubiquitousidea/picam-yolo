"""Entry point for the desktop viewer: `python -m picam_yolo.client`."""

from __future__ import annotations

import argparse
import logging
import socket
import subprocess
import sys

from .recorder import StreamRecorder
from .viewer import Viewer

log = logging.getLogger(__name__)


def _ssh_config_hostname(alias: str) -> str | None:
    """Ask ssh what `alias` expands to, for names that live only in ~/.ssh/config."""
    try:
        out = subprocess.run(
            ["ssh", "-G", alias], capture_output=True, text=True, timeout=5, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if line.startswith("hostname "):
            return line.split(None, 1)[1].strip()
    return None


def resolve_host(host: str) -> str:
    """Resolve `host`, transparently expanding SSH aliases.

    ZeroMQ's connect() is asynchronous and retries name-resolution failures
    forever without surfacing an error, so an unresolvable host looks exactly
    like a server that is not publishing. Failing loudly here turns a silent
    hang into a one-line diagnosis. SSH aliases are worth expanding because
    `--host rpi` is the natural thing to type when `ssh rpi` is how you reach
    the Pi -- but `rpi` means nothing to DNS.
    """
    try:
        socket.getaddrinfo(host, None)
        return host
    except socket.gaierror:
        pass

    real = _ssh_config_hostname(host)
    if real and real != host:
        try:
            socket.getaddrinfo(real, None)
        except socket.gaierror:
            pass
        else:
            log.info("'%s' is an SSH alias for %s", host, real)
            return real

    hint = f" ('{host}' is an SSH alias for '{real}', which also does not resolve)" if real else ""
    raise SystemExit(
        f"cannot resolve host '{host}'{hint}.\n"
        f"Pass a resolvable name or address, e.g. --host raspberrypi.local"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="picam-yolo-client",
        description="View the Pi's annotated camera streams in native windows.",
    )
    p.add_argument(
        "--host",
        default="raspberrypi.local",
        help="Pi hostname or IP (default: raspberrypi.local)",
    )
    p.add_argument("--port", type=int, default=5555)
    p.add_argument(
        "--cameras",
        default="",
        help="comma-separated camera numbers to show (empty = all)",
    )
    p.add_argument(
        "--record-dir", default="recordings", help="where clips are written (default: recordings/)"
    )
    p.add_argument(
        "--record-raw",
        action="store_true",
        help="record the clean stream without detection boxes or HUD",
    )
    p.add_argument(
        "--record-fourcc",
        default="mp4v",
        help="video codec fourcc (default: mp4v; try avc1, or MJPG with .avi)",
    )
    p.add_argument(
        "--record",
        action="store_true",
        help="start recording immediately instead of waiting for the button",
    )
    # Dog identification. Off unless --gallery is given, and everything it
    # needs (torch, the dogid package) is imported lazily, so a client with
    # only the [client] extra installed is unaffected by any of this.
    g = p.add_argument_group("dog identification (see picam_yolo.dogid)")
    g.add_argument(
        "--gallery",
        default=None,
        help="enrolled gallery npz; enables naming dogs on the stream",
    )
    g.add_argument("--embedder", choices=("hash", "torch"), default=None,
                   help="default: match the gallery")
    g.add_argument("--arch", default=None, help="default: match the gallery")
    g.add_argument("--weights", default=None, help="fine-tuned embedder checkpoint")
    g.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="override the gallery's threshold without re-enrolling",
    )
    g.add_argument("--min-margin", type=float, default=None)
    g.add_argument(
        "--identify-interval",
        type=float,
        default=0.3,
        help="minimum seconds between identification passes (default: 0.3)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    cameras = [int(x) for x in args.cameras.split(",") if x.strip()]
    host = resolve_host(args.host)
    # Bracket bare IPv6 literals for the ZeroMQ endpoint.
    host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    recorder = StreamRecorder(args.record_dir, fourcc=args.record_fourcc)
    if args.record:
        recorder.start()

    identifier = None
    if args.gallery:
        from .identity import create_identifier

        try:
            identifier = create_identifier(
                args.gallery,
                backend=args.embedder,
                arch=args.arch,
                weights=args.weights,
                min_similarity=args.min_similarity,
                min_margin=args.min_margin,
                min_interval=args.identify_interval,
            )
        except FileNotFoundError:
            raise SystemExit(
                f"no gallery at {args.gallery} -- "
                f"run `python -m picam_yolo.dogid enrol` first"
            )

    viewer = Viewer(
        endpoint=f"tcp://{host}:{args.port}",
        cameras=cameras,
        recorder=recorder,
        record_raw=args.record_raw,
        identifier=identifier,
    )
    try:
        return viewer.run()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
