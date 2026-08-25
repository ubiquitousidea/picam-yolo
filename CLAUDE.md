# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two programs from one package. `picam_yolo.server` runs on a Raspberry Pi 5: it
captures from every attached camera, runs YOLO object detection on-device, and
publishes annotated-metadata + JPEG frames over ZeroMQ. `picam_yolo.client` runs
on a desktop and draws those streams in native OpenCV windows.

They install on different machines with different dependency sets (`[server]`
and `[client]` extras) and share exactly one thing: `src/picam_yolo/protocol.py`.

## Commands

Local development and the client (this machine):

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[client]'
.venv/bin/python -m picam_yolo.client --host rpi          # view all cameras
.venv/bin/python -m picam_yolo.client --host rpi --cameras 0
```

Pi side:

```bash
HOST=rpi ./scripts/deploy.sh                              # rsync code (excludes models/)
WITH_MODELS=1 HOST=rpi ./scripts/deploy.sh                # ...and the exported model
ssh rpi 'REPO_DIR=$HOME/picam-yolo bash picam-yolo/scripts/setup_pi.sh'

./scripts/piserver.sh start --backend none                # video only, no inference
./scripts/piserver.sh start --model models/yolo11n_ncnn_model --imgsz 416
./scripts/piserver.sh log -f | status | stop

./scripts/piservice.sh stop               # stop the systemd unit (returns at boot)
./scripts/piservice.sh off                # stop and disable (stays off)
./scripts/piservice.sh on | start | restart | status
```

Exporting a model (run wherever torch is installed; the result is portable):

```bash
.venv/bin/python scripts/export_model.py --weights yolo11n.pt --imgsz 416
```

Testing without hardware — synthetic sources plus a no-op detector isolate the
wire format and viewer from cameras and inference:

```bash
python -m picam_yolo.server --synthetic 2 --backend none --bind tcp://127.0.0.1:5599
python -m picam_yolo.client --host 127.0.0.1 --port 5599
```

There is no test suite yet. `--synthetic` + `--backend none` is the current
smoke path; a real one belongs in `tests/` driving `protocol.py` round-trips and
`draw_overlay` against a synthetic frame.

## Architecture

Frames flow through one `CameraPipeline` thread per camera
(`server/pipeline.py`), each owning a `FrameSource`, its own `Detector`, and a
share of one process-wide `FramePublisher`. The pipeline is the only place that
knows the full sequence; cameras, detection, and transport don't reference each
other.

Three seams are deliberate and worth preserving:

- **`Detector` protocol** (`server/detector.py`) — `create_detector()` is a
  factory keyed by backend name. Adding Hailo or ONNX means adding a class and a
  branch there; the pipeline and CLI don't change. `NullDetector` exists to split
  "capture/network is broken" from "inference is broken" during debugging.
- **`FrameSource` protocol** (`server/cameras.py`) — `PiCameraSource` and
  `SyntheticSource` are interchangeable, which is what makes hardware-free
  development possible.
- **`protocol.py`** — the only shared module. Detection boxes are published in
  pixel coordinates *of the JPEG payload*, so the client never learns the
  inference resolution or letterbox geometry. Change this file and both halves
  must be redeployed together.

Backpressure is handled by dropping, not queueing, at both ends: the PUB socket
uses a small `SNDHWM`, and `Viewer._drain()` reads every queued message and
keeps only the newest frame per camera. A backlog would mean watching the past,
so staleness is always resolved in favour of latency.

`detect_every > 1` decouples stream rate from inference rate — every frame is
published, but detection runs on every Nth and prior boxes are reused between.
On a CPU-only Pi that trades slightly stale boxes for a much smoother preview.

## Non-obvious constraints

**Colour order is BGR everywhere.** libcamera names formats little-endian, so
Picamera2's `"RGB888"` yields a **BGR** ndarray — which is what OpenCV,
Ultralytics, and `simplejpeg(colorspace="BGR")` all want. Switching to
`"BGR888"` to "fix" this inverts every preview. See the `cameras.py` docstring.

**torch on aarch64 must come from the PyTorch CPU index.** Plain
`pip install torch` resolves to the Jetson/GH200 wheel, which declares the whole
NVIDIA CUDA stack as dependencies — gigabytes of libraries a Pi cannot use.
`scripts/setup_pi.sh` pins `--index-url https://download.pytorch.org/whl/cpu`.
Verified working: `torch-2.13.0+cpu`, Python 3.13, Debian 13 trixie.

**The Pi venv needs `--system-site-packages`.** `picamera2`, `libcamera`, and
`simplejpeg` are apt packages linked against the system libcamera; their pip
equivalents cannot open a camera. This is also why Python 3.13 (the system
interpreter) is the only usable version on the Pi.

**`--imgsz` must match the export.** NCNN bakes input dimensions into the graph.
A mismatch produces silently wrong boxes, not an error.

**One `Detector` per pipeline.** Ultralytics models are not thread-safe. The
nano weights are a few MB, so duplicating them costs far less than contention.

**`sudo` on the Pi needs a TTY.** `ssh rpi 'sudo ...'` fails with "a terminal is
required". `setup_pi.sh` therefore checks dpkg first and only escalates when a
package is genuinely missing (the desktop image already has all of them).

**Never `pkill -f <name>` over SSH when `<name>` appears in the command.**
`-f` matches full command lines, including the `bash -c` wrapper carrying it, so
pkill kills its own session — ssh dies with 255 and anything it was about to
launch never runs. The `[p]icam_yolo.server` bracket trick protects only the
pattern itself: if the plain name appears *anywhere else* in the same command
(a later `rm -f /tmp/sampler.sh`, a path, an echo), pkill matches that instead
and you are back to killing your own shell. Safest is to kill by PID from a
separate `pgrep` call, or keep the pkill in its own SSH invocation with nothing
else on the line.

**Run the server detached, and read its log separately.** Piping the server
through `ssh ... | grep` yields nothing: grep block-buffers when stdout is not a
TTY, so a run terminated by `timeout` flushes nothing. Redirect to a file on the
Pi and fetch it on a second connection:

```bash
ssh -n rpi 'cd picam-yolo && setsid .venv/bin/python -u -m picam_yolo.server ... > run.log 2>&1 < /dev/null &'
ssh -n rpi 'tail -20 picam-yolo/run.log'
```

`python -u` matters for the same buffering reason. `scripts/piserver.sh` encapsulates
all of this — stale-process kill, port-release wait, `ssh -f` launch, and a
startup confirmation that reads the log back — and should be preferred over
hand-rolled `ssh` invocations. Handing someone a raw backgrounded `ssh` command
wastes their time: it hangs their terminal while silently succeeding.

**journald is volatile here.** `journalctl --list-boots` shows only the current
boot, so post-mortems of a crash are impossible after the fact. The Pi also has
no RTC battery, so early-boot journal timestamps use a pre-NTP clock and can sit
~9 minutes behind `uptime -s`. Trust `uptime -s`.

## Hardware baseline

Raspberry Pi 5, 8 GB, Debian 13 (trixie), Python 3.13.5, kernel 6.18.34-rpt.
**No AI accelerator** — no Hailo, nothing on PCIe — so YOLO runs on the four
Cortex-A76 cores via NCNN. One `imx477` (HQ Camera) on CAM0; the code
auto-discovers, so a second module needs no code change.

The camera reports `Rotation: 180`, which Picamera2 does **not** apply
automatically. If the preview is upside down, that is why — a `libcamera.Transform`
needs threading through `PiCameraSource.start()`. Not yet implemented.

Measured, capture-only (`--backend none`, 1280x720): a steady **30.0 fps**, with
capture ~24.6 ms and JPEG encode ~8.7 ms per frame, at 39 °C. Capture plus encode
therefore already fill the 33 ms budget of a 30 fps frame interval — inference is
the limiter, and `--detect-every` is the knob that matters. **Inference
throughput has not yet been measured.**

**This board reboots under multi-core CPU load. This is the dominant
constraint on all work here.** Confirmed three times: a YOLO server run, a
4-core NCNN export, and a second YOLO run all killed it within seconds to
minutes. It does not recover on its own — it needs a manual power cycle.
`vcgencmd get_throttled` returns `0x50000` (bit 16: under-voltage occurred;
bit 18: throttling occurred). `EXT5V_V` reads a healthy 5.02 V at idle, so the
supply sags only under load. Swapping to a second power supply did not fix it.

**The workaround that works: constrain the load.** A single-core export
completed cleanly where the 4-core one crashed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 nice -n 10 .venv/bin/python scripts/export_model.py ...
```

Apply the same shape to the server (`taskset -c 0,1 --threads 1`) until the
power problem is fixed. **Inference throughput on unconstrained cores remains
unmeasured** — every attempt has crashed the board.

**Every crash zeroes recently written files.** ext4 replays its journal and
leaves 0-byte stubs. This has already destroyed an exported model (10 MB → 0 B,
four files) and `.venv/bin/pnnx`. Consequences worth remembering:

- Keep a known-good copy of `models/` on the dev machine. `WITH_MODELS=1
  ./scripts/deploy.sh` pushes it back; that is faster and safer than
  re-exporting on the Pi.
- After any crash, check for damage: `find .venv models -type f -size 0`. Beware
  false positives — `py.typed`, `REQUESTED`, and many `__init__.py` are legitimately
  empty. Executables and `dist-info/WHEEL` are not.
- `NcnnYoloDetector` now validates that `*.param`/`*.bin` are present and
  non-empty at startup, because a truncated model otherwise surfaces as
  `IndexError` deep inside Ultralytics' NCNN backend.

## Provisioning a new Pi

The repo is source-only; the exported NCNN model is a build artifact and is
deliberately not tracked. The working flow is rsync-then-provision — the Pi has
no GitHub credentials and is not expected to get any:

```bash
HOST=rpi ./scripts/deploy.sh
ssh rpi 'bash ~/picam-yolo/scripts/setup_pi.sh'
```

Cloning on the Pi works as well — the repo is public, so no credentials are
needed there — but rsync remains primary because it carries uncommitted work
that a clone cannot.

`setup_pi.sh` is idempotent and safe to re-run: it skips apt packages already
installed (the desktop image ships all of them), reuses the venv, and skips the
model export if a non-empty model is already present. Knobs: `REPO_DIR`
(default `~/picam-yolo`), `IMGSZ` (default 416, and it must match the server's
`--imgsz`), `SKIP_MODEL=1` to rsync a model in instead.

The export step is pinned to a single core on purpose — see the power section.
`scripts/deploy.sh` remains the fast path for pushing local edits to a Pi that
is already set up; it is rsync-from-dev-machine, not a provisioning tool, and
excludes `models/` unless `WITH_MODELS=1`.

## Running as a service

`scripts/install_service.sh` (run with sudo on the Pi) generates
`picam-yolo.service` from the invoking user and their checkout rather than
hardcoding `/home/pi`. Arguments come from `/etc/default/picam-yolo` via
`EnvironmentFile`, so the unit itself rarely needs editing. It sets
`CPUAffinity=0 1` for brownout protection and `StartLimitBurst=5` so a
permanent fault fails visibly instead of crash-looping.

`sudo` on this Pi requires a password and a TTY, so a system unit cannot be
installed over a non-interactive `ssh` — `ssh -t` does not help when stdin is
itself not a terminal. `scripts/install_user_service.sh` is therefore the path
that actually works here: a user unit needs no root, and `loginctl
enable-linger` succeeds unprivileged on this box, which is what makes a user
unit start at boot without a login session.

Two consequences for user units specifically: use `taskset` in `ExecStart`
rather than the `CPUAffinity=` directive (narrowing your own affinity needs no
privileges; the directive can be refused in a user manager), and set
`XDG_RUNTIME_DIR` before calling `systemctl --user` over ssh, since a
non-interactive session has no login session to point at the user manager.

`scripts/piservice.sh` drives that unit from the dev machine. It is the right
tool whenever the unit is installed: `Restart=always` means `piserver.sh stop`'s
`pkill` is undone five seconds later, so the process reappears and the stop
looks like it did nothing. `piserver.sh` now detects an active unit and stops it
properly, and refuses to `start` against one rather than losing a race for port
5555. Distinguish `stop` (unit stays enabled, returns at the next boot — which
on this board happens on its own) from `off` (`disable --now`, stays off).

Confirming a unit start means reading only the log lines *that start* appended:
the unit uses `StandardOutput=append:`, unlike `piserver.sh start` which
truncates `run.log`, so an unanchored `grep 'publishing on'` matches a stale
line from a previous run and confirms a start that in fact failed.

This Pi keeps **no user journal** — `journalctl --user -u picam-yolo` reports
"No journal files were found" — so the unit appends stdout/stderr to
`run.log`, which is also what `scripts/piserver.sh log` tails.

**`ssh -f` must have its stdout redirected.** It backgrounds the client but
keeps it alive for the life of the remote command, so an inherited pipe never
reaches EOF — `piserver.sh start | tail` hung indefinitely, printing nothing,
until `>/dev/null 2>&1` was added to the detached launch. The same trap applies
to any backgrounded ssh in a script whose output might be piped.

## Recording

`StreamRecorder` (`client/recorder.py`) opens one `cv2.VideoWriter` per camera,
lazily on first frame rather than at `start()`, because a container's frame rate
is fixed at open time and cannot be corrected afterwards.

**Derive that rate from `FrameHeader.ts`, never from local arrival times.** A
subscriber joining mid-stream is handed its whole backlog as a single TCP burst
— fifteen frames can arrive inside 60 ms, which measures as ~380 fps for a
stream genuinely running at 18. Capture timestamps are also correct when frames
are dropped: the recorded interval really is longer, and the file should play
back that way. When the viewer already has a settled estimate (the stream was
on screen before RECORD was pressed) the writer opens immediately; otherwise the
recorder buffers `PRIME_FRAMES` frames, takes the median interval, and flushes.

The record button is drawn by `draw_record_button` *after* the frame is handed
to the recorder, so UI chrome never lands in the video. `record_button_rect()`
is the single source of geometry shared by drawing and hit-testing. OpenCV's
Cocoa backend has no widget toolkit, so the button is painted into the frame and
driven by `setMouseCallback`; resizing the window can offset the hit area on
some backends, which is why `r` also toggles.

## Conventions

Both entry points are `argparse` `main(argv)` functions returning an exit code,
importable for testing. Logging goes through module-level `log =
logging.getLogger(__name__)`; the CLI configures the root logger and nothing
else calls `basicConfig`. `models/` is gitignored — it is reproducible from
`scripts/export_model.py`.
