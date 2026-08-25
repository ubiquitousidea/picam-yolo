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

**A new Pi, from the git remote** — one command, nothing else needed:

```bash
ssh rpi 'git clone git@github.com:ubiquitousidea/picam-yolo.git ~/picam-yolo && bash ~/picam-yolo/scripts/setup_pi.sh'
```

The repo is private, so the Pi needs its own credentials to clone: generate a
key on the Pi (`ssh-keygen -t ed25519`) and add the public half to GitHub as a
deploy key. If you would rather not, `scripts/deploy.sh` rsyncs from a machine
that already has the code and needs no GitHub access on the Pi at all.

That installs the apt packages, builds the venv, fetches a CPU-only torch, and
exports the NCNN model. It takes several minutes, mostly torch.

**Pushing local changes to a Pi you already set up** (assumes SSH alias `rpi`):

```bash
HOST=rpi ./scripts/deploy.sh              # code only
WITH_MODELS=1 HOST=rpi ./scripts/deploy.sh # ...and a locally exported model
```

Useful knobs for `setup_pi.sh`: `REPO_DIR` (default `~/picam-yolo`), `IMGSZ`
(default 416), and `SKIP_MODEL=1` if you intend to rsync a model over instead.
The model export is pinned to a single core deliberately — a 4-core export
browns out an under-powered Pi 5, and the reboot leaves a truncated model.

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
