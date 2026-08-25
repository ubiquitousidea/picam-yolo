#!/usr/bin/env bash
# Provision the Pi side. Run ON the Pi (or via: ssh rpi 'bash -s' < scripts/setup_pi.sh).
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/picam-yolo}"
VENV="$REPO_DIR/.venv"

# picamera2 and simplejpeg come from apt, never pip: the pip builds do not link
# against the system libcamera and will fail to open a camera.
APT_PKGS=(python3-picamera2 python3-simplejpeg python3-numpy python3-venv rpicam-apps)
MISSING=()
for pkg in "${APT_PKGS[@]}"; do
  dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "^install ok installed$" || MISSING+=("$pkg")
done

if (( ${#MISSING[@]} )); then
  # Only reached on Lite images; the desktop image ships all of these, and
  # sudo here needs a TTY, which breaks `ssh rpi 'bash -s' < this-script`.
  echo "==> apt packages: ${MISSING[*]}"
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends "${MISSING[@]}"
else
  echo "==> apt packages already present, skipping"
fi

echo "==> venv at $VENV"
# --system-site-packages is required, not a convenience: it is how the venv sees
# the apt-installed picamera2/libcamera bindings, which cannot be pip-installed.
python3 -m venv --system-site-packages "$VENV"

echo "==> torch (CPU-only build)"
"$VENV/bin/pip" install --upgrade pip
# Must come from the PyTorch CPU index. On aarch64, plain `pip install torch`
# resolves to the Jetson/GH200 wheel, which declares the whole NVIDIA CUDA stack
# as dependencies -- several GB of libraries a Pi can never use.
"$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch torchvision

echo "==> server dependencies"
"$VENV/bin/pip" install -e "$REPO_DIR[server]"

# --- model ------------------------------------------------------------------
# The exported NCNN model is a build artifact, so a fresh clone will not have
# one. Export it here unless the caller supplied it another way.
MODEL_DIR="$REPO_DIR/models/yolo11n_ncnn_model"
IMGSZ="${IMGSZ:-416}"

if [[ "${SKIP_MODEL:-0}" == "1" ]]; then
  echo "==> skipping model export (SKIP_MODEL=1)"
elif [[ -s "$MODEL_DIR/model.ncnn.param" && -s "$MODEL_DIR/model.ncnn.bin" ]]; then
  echo "==> model already present at $MODEL_DIR"
else
  # Pinned to one core on purpose. A 4-core export browns out an
  # under-powered Pi 5, and the resulting hard reboot leaves the half-written
  # model as 0-byte files. Single-core is slower but has completed reliably.
  echo "==> exporting NCNN model at imgsz=$IMGSZ (single core; takes a few minutes)"
  ( cd "$REPO_DIR" && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      taskset -c 0 nice -n 10 "$VENV/bin/python" scripts/export_model.py --imgsz "$IMGSZ" )
  sync   # flush before anything else can trigger a brownout
fi

echo
echo "done. Verify with:"
echo "  $VENV/bin/python -c 'from picamera2 import Picamera2; print(Picamera2.global_camera_info())'"
echo
echo "Run the server:"
echo "  cd $REPO_DIR && .venv/bin/python -m picam_yolo.server --model models/yolo11n_ncnn_model --imgsz $IMGSZ"
