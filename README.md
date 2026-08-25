# picam-yolo

Capture from every camera attached to a Raspberry Pi 5, run YOLO object
detection on the Pi, and view the annotated streams in native windows on your
desktop.

```
Pi 5                                          Desktop
┌────────────────────────────────────┐        ┌─────────────────────┐
│ Picamera2 ─▶ YOLO/NCNN ─▶ JPEG ─────┼─ ZMQ ─▶│ decode ─▶ overlay   │
│   (one thread per camera)          │  PUB   │   (one window/cam)  │
└────────────────────────────────────┘        └─────────────────────┘
```

Detections travel as JSON alongside the JPEG, in payload pixel coordinates, so
boxes are drawn client-side and stay crisp at any window size.

## Setup

**Desktop:**

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[client]'
```

**A new Pi** — rsync the code over, then provision it in place:

```bash
HOST=rpi ./scripts/deploy.sh
ssh rpi 'bash ~/picam-yolo/scripts/setup_pi.sh'
```

`setup_pi.sh` installs the apt packages, builds the venv with
`--system-site-packages`, fetches a CPU-only torch, and exports the NCNN model.
It takes several minutes, mostly torch. It is idempotent, so re-running it after
a later `deploy.sh` is harmless.

**Pushing later changes** to a Pi that is already set up:

```bash
HOST=rpi ./scripts/deploy.sh               # code only
WITH_MODELS=1 HOST=rpi ./scripts/deploy.sh # ...and a locally exported model
```

Knobs for `setup_pi.sh`: `REPO_DIR` (default `~/picam-yolo`), `IMGSZ` (default
416, and it must match the server's `--imgsz`), `SKIP_MODEL=1` when you intend
to rsync a model over instead. The model export is pinned to a single core
deliberately — a 4-core export browns out an under-powered Pi 5, and the reboot
leaves a truncated model behind.

Cloning this repo directly on the Pi works too, and needs no credentials now
that it is public:

```bash
ssh rpi 'git clone https://github.com/ubiquitousidea/picam-yolo.git ~/picam-yolo && bash ~/picam-yolo/scripts/setup_pi.sh'
```

`deploy.sh` stays the primary path because it pushes uncommitted work in
progress, which a clone cannot.

## Run

```bash
# start the Pi server (stops any stale one, waits for the port, confirms startup)
./scripts/piserver.sh start --backend none --size 1280x720          # video only
./scripts/piserver.sh start --model models/yolo11n_ncnn_model --imgsz 416   # with YOLO

# watch it
./scripts/piserver.sh log -f
./scripts/piserver.sh status
./scripts/piserver.sh stop

# on the desktop
.venv/bin/python -m picam_yolo.client --host rpi
```

Use `piserver.sh` rather than a raw `ssh ... &`: a backgrounded remote command
can hold the SSH channel open and hang your terminal with no output, which looks
identical to a failure while the server is actually running fine.

`q` or `Esc` quits, `h` toggles the stats overlay.

## Recording

Click **RECORD** in the top-right of the window, or press `r`. Clips land in
`recordings/` as `cam<N>_<timestamp>.mp4`, one file per camera, and the button
chrome is never baked into the video.

```bash
--record            # start recording immediately at launch
--record-raw        # clean stream, without detection boxes or the HUD
--record-dir DIR    # default: recordings/
--record-fourcc X   # default mp4v; try avc1, or MJPG with a .avi extension
```

The file's frame rate is measured from the publisher's capture timestamps, so a
clip plays back at true speed even though the stream rate varies with inference
load. Recording stops and finalises the file on quit or Ctrl-C.

## Run at boot

```bash
ssh rpi 'bash ~/picam-yolo/scripts/install_user_service.sh'
```

A **user** service, so it needs no root — handy on a Pi where `sudo` wants a
password and a TTY. It enables lingering so the unit starts at boot without a
login session.

```bash
nano ~/.config/picam-yolo.env          # e.g. add --detect-every 2
systemctl --user restart picam-yolo
tail -f ~/picam-yolo/run.log           # live log
systemctl --user disable --now picam-yolo
```

Where passwordless sudo is available, `scripts/install_service.sh` installs the
equivalent system unit instead (config in `/etc/default/picam-yolo`).

Both pin the server to two cores for the reason given under Tuning, and give up
after repeated failed starts rather than crash-looping on a permanent fault such
as a missing model. Output goes to `run.log` rather than the journal, because
this Pi keeps no user journal; the file grows unbounded, so truncate it if that
ever matters.

## Tuning

CPU-only inference is the bottleneck. In rough order of effect:

| Change | Effect |
|---|---|
| `--detect-every 2` | Publishes every frame, detects on every other one. Big FPS win, slightly stale boxes. |
| `--imgsz 320` (re-export to match) | Faster inference, weaker on small/distant objects. |
| `--size 960x540` | Cheaper capture and JPEG encode. |
| `--classes 0` | Person only. Filters during NMS. |
| `--jpeg-quality 65` | Less bandwidth, mild artifacts. |

`--backend none` disables detection entirely, which is the quickest way to tell
whether a problem is in inference or in capture/transport.

## No hardware?

```bash
python -m picam_yolo.server --synthetic 2 --backend none --bind tcp://127.0.0.1:5599
python -m picam_yolo.client --host 127.0.0.1 --port 5599
```

Two test-pattern streams, no cameras and no model required.
