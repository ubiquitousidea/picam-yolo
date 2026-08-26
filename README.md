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

`scripts/piservice.sh` drives that unit from the dev machine, so idling the Pi
does not need an interactive login:

```bash
./scripts/piservice.sh stop            # stop now -- returns at the next boot
./scripts/piservice.sh off             # stop *and* disable -- stays off
./scripts/piservice.sh on              # enable and start again
./scripts/piservice.sh status          # active / enabled / pid
```

Stopping the unit is what actually drops the Pi's power draw: capture and NCNN
inference are what keep two cores busy, and both end with the process. Note the
difference between `stop` and `off` — a stopped-but-enabled unit comes back at
the next boot, which on this board can happen on its own.

Where passwordless sudo is available, `scripts/install_service.sh` installs the
equivalent system unit instead (config in `/etc/default/picam-yolo`).

Both pin the server to two cores for the reason given under Tuning, and give up
after repeated failed starts rather than crash-looping on a permanent fault such
as a missing model. Output goes to `run.log` rather than the journal, because
this Pi keeps no user journal; the file grows unbounded, so truncate it if that
ever matters.

## Identifying individual dogs

The Pi's YOLO answers "there is a dog". `picam_yolo.dogid` answers "that is Rex".

It does **not** retrain the detector. A second stage embeds the dog crop and
matches it against a gallery of enrolled dogs, so adding a new dog is enrolling
~20 crops rather than a training run, and the Pi's NCNN model never changes.
Everything here runs on the desktop -- the board has neither the power budget
nor any need for a labelling GUI.

```bash
pip install -e '.[dogid]'

python -m picam_yolo.dogid capture --host rpi --seconds 300   # harvest crops
python -m picam_yolo.dogid label                              # assign names
python -m picam_yolo.dogid enrol --embedder torch             # build the gallery
python -m picam_yolo.dogid eval --suggest-threshold           # score it, tune the gates
python -m picam_yolo.dogid stats

# then watch it work on the live stream
python -m picam_yolo.client --host rpi --gallery dogid/gallery.npz
```

Shared options (`--embedder`, `--root`, `--gallery`, `-v`) work on either side
of the subcommand.

### Capture

`capture` costs the Pi nothing: the boxes already travel in `FrameHeader`, so it
subscribes to the stream the server is already publishing and cuts out the crops
YOLO already found. Near-duplicate frames are dropped by perceptual hash -- a
sleeping dog otherwise contributes hundreds of identical crops that make the
gallery confident about exactly one pose.

**Shutter speed matters more than anything else here.** A moving dog under an
unconstrained auto-exposure is a smear, and no embedder recovers identity from a
smear. Run supervised sessions with a fixed short exposure:

```bash
./scripts/piserver.sh start --model models/yolo11n_ncnn_model --imgsz 416 \
    --size 2028x1520 --detect-every 2 --jpeg-quality 60 \
    --exposure-us 5000 --gain 12
```

Measured on a 55-90 lux indoor scene, that took crop sharpness (variance of
Laplacian) from a median of 8 to 29, and the share of usable crops from 4% to
76%, at the same brightness. See [Camera exposure](#camera-exposure).

### Label

`label` proposes and you confirm: once any gallery exists each crop arrives with
a suggested name and Enter accepts it, so only the disagreements cost attention.

Keys: digits assign an enrolled dog, `n` types a new name, `u` marks an
unfamiliar dog, `x` marks a bad detection, `m` discards an unusable crop, `s`
skips, `b` steps back.

**`m` is not the same as `u`/`x`.** Use it when a crop cannot be used at all --
two dogs inside one padded box, a motion smear, a dog half out of frame. Those
are excluded from the dog centroids *and* from the rejection class. Sending them
to `u`/`x` instead would enrol them as negatives, teaching the gallery that your
own dogs resemble the class whose job is to turn strangers away. `s` also keeps
a crop out of the gallery, but leaves it unlabelled, so every later session
offers it again; `m` records the judgement once.

**Ordering inverts once a gallery exists**, and `auto` handles it:

| state | order | why |
|---|---|---|
| no gallery yet | most-confident first | clean full-body shots make good centroids |
| gallery exists | least-certain first | those crops move the boundary |

Getting that backwards is expensive. Before this was fixed, the first pass
showed the 40 *least* confident crops of 1505 -- dark, part-occluded, several
holding two overlapping dogs -- and every centroid would have been built from
them. Override with `--order confident|uncertain|captured`.

Embeddings are cached by crop id, and crops are content-addressed, so a cached
vector can never go stale. Ranking 1470 crops -- which `label` does at startup,
since choosing the most uncertain N means scoring all of them -- went from 120 s
to 0.01 s.

### Enrol and evaluate

`enrol` builds one centroid per dog and reports **cohesion**, the mean similarity
of a dog's crops to their own centroid. Below ~0.6 usually means a mislabel
slipped in rather than difficult lighting.

`eval` scores held-out crops (every 5th labelled crop goes to validation) and
prints a confusion matrix. Read the two failure modes separately, because they
have opposite fixes:

- **Off-diagonal entries** -- one dog named as another. That wants a better
  embedder, and is the only thing that justifies `finetune`.
- **Rejections** -- a known dog the gallery declined to name. That wants looser
  gates, and `eval` names which gate did it.

**The margin gate is usually the one that binds.** Cosine similarity to the
nearest centroid runs high for *any* dog, so `min_similarity` rarely rejects
anything, while `min_margin` -- how far the best match must beat the runner-up --
does. On a real two-dog gallery, `min_similarity=0.55` rejected nobody (every
similarity was 0.774-0.920) while `min_margin=0.05` rejected 8 of 11 validation
crops. Dropping the margin to 0.01 moved accuracy from **27.3% to 90.9% with
nothing rejected**, changing nothing else:

```bash
python -m picam_yolo.dogid enrol --embedder torch --min-margin 0.01
```

`--suggest-threshold` sweeps both gates and reports the pair that scores best,
and warns when the validation split is too small to trust. Be sceptical of a
suggested `min_similarity` fitted to a handful of crops: on that same gallery it
proposed a value that rejected 9% more for no accuracy gain.

**Treat a small validation split as a hint.** The same gallery scored 90.9% on
11 val crops and 64.7% once there were 17 -- the extra crops were not harder,
there were simply enough of them to show a real confusion the first sample had
missed. Label more before believing a number.

Re-running `enrol` keeps the gates from the gallery it replaces, since they are
the tuned part and live nowhere else; pass `--min-margin`/`--min-similarity`
explicitly to change them. Without that, rebuilding after labelling more crops
quietly reverted a tuned 0.01 margin to the 0.05 default.

Label some `u`/`x` crops too. They become a `__reject__` centroid, and without
one the gallery has nothing to compare a stranger against -- it will name an
unfamiliar dog as one of yours.

### Live identification

```bash
python -m picam_yolo.client --host rpi --gallery dogid/gallery.npz
python -m picam_yolo.client --host rpi --gallery dogid/gallery.npz --min-margin 0.02
```

Names are drawn on the boxes, one stable colour per dog; `i` toggles it. An
unrecognised dog renders as `dog ? 0.42` with the similarity that fell short,
so you can judge a threshold while standing in front of the camera.
`--min-similarity` / `--min-margin` override the values baked into the gallery
without re-enrolling.

Identification runs on a worker thread and never blocks the preview. It has to:
embedding costs ~97 ms for one crop against the ~52 ms between frames at 19 fps.
The worker keeps only the newest pending frame per camera, and the render thread
draws the last result mapped onto current boxes by IoU -- identity is stable over
a few hundred milliseconds in a way that box position is not. Measured against
the live stream, the render step stays at 0.1 ms median and the viewer holds
full stream rate.

`--gallery` is what switches this on; torch and the `dogid` package are imported
lazily, so a client installed with only `[client]` is unaffected.

### Embedders

`--embedder hash` needs no torch and exists to smoke-test the pipeline without
a model, the way `--backend none` does on the server. It is not good enough to
tell two brown terriers apart; use `--embedder torch` for real work.

A gallery records which embedder built it, and `label`/`eval` default to that
one -- centroids are meaningless under any other backbone. Re-run `enrol` after
changing backbone or weights.

Start with `enrol` on a pretrained backbone and read `eval`. ImageNet features
separate visually distinct dogs well enough that no training may be needed at
all. `finetune` (a skeleton) is only worth building when the confusion matrix
shows two specific dogs bleeding into each other.

## Tuning

CPU-only inference is the bottleneck. In rough order of effect:

| Change | Effect |
|---|---|
| `--detect-every 2` | Publishes every frame, detects on every other one. Big FPS win, slightly stale boxes. |
| `--imgsz 320` (re-export to match) | Faster inference, weaker on small/distant objects. |
| `--size 960x540` | Cheaper capture and JPEG encode. |
| `--classes 0` | Person only. Filters during NMS. |
| `--jpeg-quality 65` | Less bandwidth, mild artifacts. Pair with `--detect-every`. |
| `--exposure-mode short` | Shorter shutter, less motion blur. See [Camera exposure](#camera-exposure). |

`--backend none` disables detection entirely, which is the quickest way to tell
whether a problem is in inference or in capture/transport.

Server-side frame rate is not delivered frame rate. The link here tops out
around 29 Mbit/s, and it is a *byte* budget: raising `--detect-every` without
also lowering `--jpeg-quality` produces extra frames that are only dropped in
flight. Compare `FrameHeader.seq` spans against frames received to see it --
rate alone hides drops, and a short sample reads far too high because a
subscriber is handed its backlog as a burst on join.

## Camera exposure

With no controls set, auto-exposure picks whatever shutter it likes, and indoors
it picks a long one. Anything moving is then smeared -- which does not matter
much for "is there a dog" but ruins "which dog".

```bash
--exposure-mode short     # bias AE toward a fast shutter and higher gain
--metering spot|centre|matrix
--ev 1.0                  # exposure compensation, in stops
--exposure-us 5000        # pin the shutter; disables AE (use with --gain)
--gain 12                 # analogue gain
```

Measured on one camera and scene at 2028x1520, 55-90 lux:

| setting | exposure | gain | brightness | sharpness |
|---|---|---|---|---|
| none (AE unconstrained) | 33.0 ms | 2.14 | — | — |
| `--exposure-mode short` | 20.0 ms | 3.56 | 101/255 | 41.1 |
| `--exposure-mode short --ev 0.7` | 33.0 ms | 4.00 | — | — |
| `--exposure-us 5000 --gain 12` | 5.0 ms | 11.91 | 92/255 | 40.8 |
| `--exposure-us 2000 --gain 22` | 2.0 ms | 21.79 | 77/255 | 34.9 |

Two things worth knowing:

- **`--ev` cancels `--exposure-mode short`.** EV compensation asks AE for a
  brighter picture and AE buys it with a *longer* shutter, putting 20 ms back to
  33 ms. For a subject underexposed against a bright background reach for
  `--metering spot` instead.
- **5 ms at gain 12 is the sweet spot** on this camera: it holds brightness and
  sharpness against the 20 ms result while cutting the shutter 4x, so gain noise
  costs nothing measurable. Below that, brightness falls away and gain hits the
  sensor ceiling.

Use `--exposure-mode short` for the long-running service, where AE stays in
charge and survives the room getting darker. Use a pinned `--exposure-us` only
for supervised sessions -- left in the unit, the first dark evening gives you a
black stream.

The server logs what AE settled on at startup, which is the number to check
first when crops come back blurry:

```
cam0 (imx477) AE settled: exposure 5.0 ms, analogue gain 11.91, lux 90
```

## No hardware?

```bash
python -m picam_yolo.server --synthetic 2 --backend none --bind tcp://127.0.0.1:5599
python -m picam_yolo.client --host 127.0.0.1 --port 5599
```

Two test-pattern streams, no cameras and no model required.
