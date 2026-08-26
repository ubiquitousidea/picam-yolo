# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Three programs from one package. `picam_yolo.server` runs on a Raspberry Pi 5:
it captures from every attached camera, runs YOLO object detection on-device,
and publishes annotated-metadata + JPEG frames over ZeroMQ. `picam_yolo.client`
runs on a desktop and draws those streams in native OpenCV windows.
`picam_yolo.dogid` is a desktop-only workflow that identifies *which* dog, on
top of the detector that already finds dogs.

They install on different machines with different dependency sets (`[server]`,
`[client]`, `[dogid]`). Server and client share exactly one thing:
`src/picam_yolo/protocol.py`. `dogid` reads that wire format but never changes
it — see its section below.

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

Dog re-identification (desktop only; see `src/picam_yolo/dogid/`):

```bash
python -m picam_yolo.dogid capture --host rpi --seconds 120
python -m picam_yolo.dogid label                    # OpenCV GUI, AI-suggested
python -m picam_yolo.dogid enrol --embedder torch
python -m picam_yolo.dogid eval  --embedder torch --suggest-threshold
python -m pytest tests/                             # no torch, no camera, no Pi

# ...then watch it work on the live stream:
python -m picam_yolo.client --host rpi --gallery dogid/gallery.npz
python -m picam_yolo.client --host rpi --gallery dogid/gallery.npz --min-similarity 0.45
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

`tests/` covers `dogid` and the viewer's identity path (`pytest tests/`, no
torch or hardware required). `--synthetic` + `--backend none` remains the smoke
path for the server and client; `protocol.py` round-trips are still untested and
are the obvious next addition.

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
- **`Embedder` / `Gallery`** (`dogid/embed.py`, `dogid/gallery.py`) — the same
  Protocol-plus-factory shape as `Detector`. `HashEmbedder` is the `NullDetector`
  of this subsystem: no torch, no weights, so the dataset, labeller and CLI can
  be exercised with nothing installed.
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

## Exposure, and why crops came back blurry

The first dog-id capture session produced 233 crops that were all unusable:
sharpness (variance of Laplacian) measured **median 8, max 35** against **36 for
the static background** — the subject was smeared while the lens was perfectly in
focus. `PiCameraSource.start()` set no camera controls at all, so AE was free to
pick a long shutter, and in a 55–90 lux room it did.

Measured on this camera, same scene, at 2028x1520:

| setting | exposure | gain | frame brightness | frame sharpness |
|---|---|---|---|---|
| (none — old default) | 33.0 ms | 2.14 | — | — |
| `--exposure-mode short` | **20.0 ms** | 3.56 | 101/255 | 41.1 |
| `--exposure-mode short --ev 0.7` | 33.0 ms | 4.00 | — | — |
| `--exposure-us 8000 --gain 8` | 8.0 ms | 8.00 | 93.6/255 | 40.5 |
| `--exposure-us 5000 --gain 12` | **5.0 ms** | 11.91 | 92.0/255 | 40.8 |
| `--exposure-us 3000 --gain 16` | 3.0 ms | 16.00 | 81.2/255 | 36.2 |
| `--exposure-us 2000 --gain 22` | 2.0 ms | 21.79 | 77.0/255 | 34.9 |

Three things this settles:

- **`--ev` cancels `--exposure-mode short`.** EV compensation asks AE for a
  brighter picture, and AE spends the budget on a *longer* shutter — 20 ms went
  back to 33 ms. The combination is the worst of both worlds. If a subject is
  underexposed against a bright background, reach for `--metering spot`, not
  `--ev`.
- **5 ms at gain 12 is the sweet spot.** It holds brightness (92 vs 101) and
  static sharpness (40.8 vs 41.1) against the 20 ms AE result while cutting the
  shutter 4x — so 4x less motion blur, and the gain noise costs nothing
  measurable. Below 5 ms brightness falls away and gain hits the sensor ceiling
  (21.79 at 2 ms).
- **There is no AE-preserving hard cap on exposure.** `FrameDurationLimits` would
  bound it, but its maximum also sets a *minimum* frame rate, and 8 ms implies
  125 fps, which the sensor cannot deliver at this resolution. Short of manual,
  `--exposure-mode short` is the only lever.

So: `--exposure-mode short` is the right **service** default — AE stays in charge
and survives the room getting darker. `--exposure-us 5000 --gain 12` is for a
**supervised capture session**, where a fixed shutter is safe because someone is
watching; leave it in the unit and the first cloudy evening produces a black
stream. Same rule as the four-core experiment: don't park a fixed config in the
unit.

**Sharpness is measured, not eyeballed.** Variance of Laplacian on the crop,
compared against the same statistic on the static background of the same frame.
The absolute scale is meaningless here (q60 JPEG smooths high frequencies); the
*ratio* to the static scene is what says whether the subject is blurred or the
lens is off. Beware two confounds when comparing sessions: the metric rises with
brightness, so an EV change inflates it on its own, and a dog standing still is
sharp at any exposure.

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
therefore already fill the 33 ms budget of a 30 fps frame interval, and
`--detect-every` is the knob that matters.

Measured 2026-08-25 at **full sensor resolution** (`--size 4056x3040 --imgsz 416`,
NCNN, pinned to cores 0,1 by the user unit): a steady **6.2–6.3 fps** over four
minutes at 58 °C, `throttled=0x0`, no reboot. Per frame: capture ~13 ms, **infer
~52 ms**, encode ~93 ms. Two things this settles:

- **Inference is ~52 ms/frame** on two pinned cores at imgsz 416 — about 19 fps
  if it ran alone. At full resolution it is *not* the limiter; **JPEG encode is**,
  at ~93 ms. Encode scales with pixel count, inference does not (the frame is
  letterboxed to `--imgsz` regardless).
- Capture *time* falls with resolution (24.6 ms → 13 ms) because `capture_array`
  mostly blocks waiting for the next frame. At 6 fps a frame is already waiting,
  so the number measures idle time, not data volume. Do not read it as "bigger
  frames capture faster".

Measured the same day at **2028x1520** (3.08 MP, a native 2x2-binned mode, so
the ISP does no rescaling and the full 4:3 FOV is kept) — the current setting:
a steady **12.9 fps** at 58 °C, `throttled=0x0`. Per frame: capture ~3.8 ms,
infer ~49 ms, encode ~23.5 ms. Quartering the pixels roughly quartered the
encode and doubled the frame rate, exactly as the pixel-count model predicts,
and **inference is the limiter again** at ~49 of the ~77 ms budget. From here
`--detect-every 2` is the next real gain, not a smaller frame: it lifts the
server to **19.1 fps** (inference every other frame, boxes reused between), still
on two cores and still `throttled=0x0`.

**But server fps is not delivered fps.** At q80 that 19 fps stream needs ~40
Mbit/s and the link gives ~29, so 28 % of frames were dropped and the client saw
13.1 -- no better than before. Adding `--jpeg-quality 60` cuts frames from ~270
to ~190 KiB, which fits, and the client then receives **18.8 fps with zero gaps**.
The pairing is the point: `--detect-every` and `--jpeg-quality` must move
together, or the extra frames are generated only to be discarded in flight.

**The wifi link tops out at ~29 Mbit/s, and that binds before CPU does.**
Two independent saturated runs measured 28.9 and 29.2 Mbit/s. It is a *byte*
budget, not a frame budget: delivered fps is roughly `29 Mbit/s / bytes-per-frame`,
whatever the server manages to publish. Server-side fps above that line converts
entirely into dropped frames.

**Measure delivered rate over a window, after a warmup.** A subscriber is handed
its backlog as a burst on join, so a short sample reads far too high -- one
8-frame sample of this stream showed "48.8 Mbit/s, no gaps" where a 9-second
sample of the *same* stream showed 28.9 Mbit/s and **28 % of frames missing**.
Same trap `client/recorder.py` documents for frame-rate estimation. Compare
`FrameHeader.seq` spans against the received count; rate alone hides drops.

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

**The supply is the root cause, and it is now measured.** The powerbank in use
on 2026-08-25 advertises these USB-PD profiles: 5 V/2.4 A (12 W), 9 V/2.22 A,
12 V/1.67 A, PPS 3.3-11 V/2 A. **The Pi 5 draws only from the 5 V rail**, so the
bank's headline 20 W is unreachable and the real budget is **12 W** -- under half
the 25 W an official Pi 5 supply provides. Firmware agrees:
`/sys/firmware/devicetree/base/chosen/power/max_current` reads 3000 (the
conservative fallback) and `usb_max_current_enable` is 0. A supply that
negotiates 5 V/5 A makes `max_current` read 5000; that is the check worth
running on any replacement.

Measured draw on two cores: board total ~5.4 W, of which VDD_CORE is 2.7 W
median but **5.4 W peak** -- brownout is driven by the peak, not the average.
Doubling the core term projects ~2.9 A at 5 V against a 2.4 A limit.

**Four cores was tried on 2026-08-25 and rebooted the board in under a minute**,
confirming that projection. Two findings from it:

- It is not worth retrying even with a bigger supply. The one surviving 4-core
  sample managed **15.4 fps against 12.9 on two cores -- about 20 %** -- because
  NCNN already saturates two cores at imgsz 416. `--detect-every 2` beats that
  comfortably on two cores at no stability cost.
- **Never park an experimental config in the unit.** The drop-in that enabled
  four cores survived the reboot it caused, and the enabled unit auto-started
  straight back into it -- a crash loop, caught only by hand. Test risky
  settings with a one-shot foreground run, not a persistent unit change.

Two-core pinning otherwise holds: four minutes at full sensor resolution kept
`throttled=0x0` at 58 C. Note that `get_throttled` resets at boot, so a crash
leaves no forensic trace -- the `0x0` seen after a reboot means nothing.

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

## Identifying individual dogs

`picam_yolo.dogid` is a third component, installed on the **desktop only**
(`[dogid]` extra). Two decisions shape it, and both were deliberate:

**Two-stage, not a retrained detector.** The Pi's YOLO keeps answering "dog";
a separate embedder maps the crop to a vector and a gallery of per-dog centroids
names it. Folding per-dog classes into the detection head would mean a full
retrain plus NCNN re-export for every new dog, hundreds of boxes each, and
fine-grained identity from a nano model that sees a distant dog as ~60 px.
Enrolment is instead arithmetic over ~20 crops, and `models/` never changes.

**Identification runs on the client, not the Pi.** The client already receives
the JPEG and the boxes, so it can crop and identify locally. `protocol.py` is
untouched — no paired redeploy — and the board, which browns out at four cores
and already spends ~50 ms of its frame budget on inference, does no extra work.
Putting identity on the wire would need a `Detection.identity` field and a
simultaneous redeploy of both halves; that is the seam if it ever becomes worth
it.

**Identification is slower than the frame interval, and that shapes
`client/identity.py`.** Measured on the dev machine, `mobilenet_v3_small` costs
~97 ms for one crop and ~195 ms for four, against the ~52 ms between frames at
19 fps; on real 2028x1520 frames a one-crop pass measured ~120 ms. So it cannot
run on the render thread, and it does not:

- A worker thread embeds. `submit()` never blocks and keeps only the **newest**
  pending frame per camera, discarding anything queued behind it — the same
  drop-don't-queue rule as `Viewer._drain()` and the PUB socket. A backlog here
  would mean naming a dog from a frame that is seconds old.
- The render thread draws the last result, mapped onto the *current* boxes by
  IoU (`IdentityTracker`). Identity is stable over a few hundred milliseconds in
  a way that box position is not. Measured against the live stream: the render
  step stays at **0.1 ms median** with identification on, and the viewer holds
  full stream rate.
- Results are tracked **per camera**. One shared result set would let cam0's dogs
  claim cam1's boxes — the IoU test compares coordinates, which say nothing about
  which camera they came from.
- Results expire after `MAX_AGE_S`, or a name outlives the dog that earned it and
  lands on whatever walks through that patch of frame next.

**Query-time crops must match enrolment-time crops.** `identify_frame` cuts with
the same `pad_box`, `PAD_FRAC` and `MIN_CROP_PX` that `dogid.capture` used to
build the gallery. Cropping tighter or looser at query time shifts the embedding
distribution away from the enrolled one, which surfaces as mysteriously low
similarity rather than as an error. The same trap caught the tests: a fixture
enrolling bare patches scored 0.46 against query crops carrying `pad_box`'s 12 %
of background, so the fixture now enrols through `CropHarvester` like the real
thing does.

**A rejection is drawn, not hidden.** An unrecognised dog renders as `dog ? 0.42`
with the similarity that fell short — the feedback that says whether to collect
more crops or just lower the threshold, visible while you are still standing in
front of the camera. `--min-similarity` / `--min-margin` override the values
baked into the npz, because re-enrolling to try a number re-embeds the whole
dataset. `i` toggles identification; each name gets a stable colour from crc32,
not `hash()`, which PYTHONHASHSEED randomises per process.

`--gallery` switches all of this on. Everything it needs — torch, the `dogid`
package — is imported lazily inside `create_identifier`, so a client installed
with only the `[client]` extra imports neither and is unaffected.

Non-obvious bits:

- **Crops are content-addressed by SHA-1 of their JPEG bytes**, so re-running
  `capture` is idempotent. The manifest is append-only JSONL and `load()` folds
  it so later records win; a truncated last line is skipped rather than fatal,
  because interrupting a long labelling session should not cost the manifest.
- **Near-duplicate crops are dropped by dhash.** A sleeping dog otherwise
  contributes hundreds of identical frames and the gallery becomes confident
  about exactly one pose.
- **Labelling order inverts once a gallery exists.** Before one, show the
  *most* confident crops: they are the clean full-body shots the first centroids
  get built from. After one, show the least certain, which are the crops that
  move the boundary. `order_for_labelling` picks on `auto`; getting it backwards
  is costly, since everything downstream rests on those first centroids.
- **"Unusable" and "negative" are different labels, and conflating them poisons
  the gallery.** `m` writes `__discard__` for a crop that cannot be used: two
  dogs inside one padded box, a motion smear, a dog half out of frame. It is
  excluded from the identities *and* from the rejection class. Sending such a
  crop to `u`/`x` instead would enrol it in `__reject__` — teaching the gallery
  that our own dogs resemble the class whose whole job is to turn strangers
  down. That is why `RESERVED` (not an identity) and `NEGATIVE_LABELS` (seeds
  the rejection class) are separate sets in `dataset.py`. `s` also keeps a crop
  out of the gallery, but leaves it unlabelled, so every future `label` session
  offers it again; `m` records the judgement once.
- **`min_margin` matters as much as `min_similarity`.** Cosine similarity to the
  nearest centroid runs high for *any* dog, so an absolute threshold alone
  confidently misnames strangers. `__unknown__` and `__not_a_dog__` are enrolled
  as a rejection class for the same reason — they are what turn "the detector
  saw a cat" into a non-answer instead of a wrong answer.
- **An accuracy assertion cannot detect a collapsed embedding space.** The first
  `HashEmbedder` averaged H, S and V together, which put every crop within 0.001
  cosine of every other; argmax was noise, yet the round-trip test still passed.
  `test_embeddings_separate_different_dogs` asserts within-class vs
  between-class separation directly, which is the property that actually
  matters. Keep that test whenever the embedder changes.
- **The gallery is embedder-specific.** Re-run `enrol` after any change of
  backbone or weights; centroids built under the old weights are meaningless
  under the new ones.

`train.finetune()` is a deliberate skeleton — its docstring carries the intended
shape. Run `eval` first: a pretrained backbone plus `enrol` is often enough, and
metric learning is only worth it when the confusion matrix shows two dogs
actually bleeding into each other.

## Conventions

Both entry points are `argparse` `main(argv)` functions returning an exit code,
importable for testing. Logging goes through module-level `log =
logging.getLogger(__name__)`; the CLI configures the root logger and nothing
else calls `basicConfig`. `models/` is gitignored — it is reproducible from
`scripts/export_model.py`.
