#!/usr/bin/env bash
# Install picam-yolo as a systemd service so it starts at boot.
# Run ON the Pi, with sudo:   sudo bash ~/picam-yolo/scripts/install_service.sh
set -euo pipefail

UNIT=/etc/systemd/system/picam-yolo.service
DEFAULTS=/etc/default/picam-yolo

if [[ $EUID -ne 0 ]]; then
  echo "must run as root: sudo bash $0" >&2
  exit 1
fi

# Resolve the invoking user and their checkout, so this works for any user and
# any REPO_DIR rather than hardcoding /home/pi.
SVC_USER="${SUDO_USER:-pi}"
REPO_DIR="${REPO_DIR:-$(getent passwd "$SVC_USER" | cut -d: -f6)/picam-yolo}"
VENV_PY="$REPO_DIR/.venv/bin/python"
# Cores the server may use, and the inference threads to match. Four browns out
# a board fed by an under-spec supply, so check before widening:
#   od -An -tu4 --endian=big /sys/firmware/devicetree/base/chosen/power/max_current
# 5000 means the supply negotiated 5V/5A and four cores are affordable; 3000 is
# the conservative fallback, and CORES="0 1" THREADS=1 is the safe setting there.
CORES="${CORES:-0 1 2 3}"
THREADS="${THREADS:-4}"

[[ -x "$VENV_PY" ]] || { echo "no venv at $VENV_PY -- run scripts/setup_pi.sh first" >&2; exit 1; }

# A manually launched server would hold the PUB port and make the service fail
# its first start for a reason that looks nothing like the real cause.
pkill -f '[p]icam_yolo.server' 2>/dev/null || true

if [[ ! -f "$DEFAULTS" ]]; then
  cat > "$DEFAULTS" <<EOF
# Arguments for picam-yolo.service. Edit, then: sudo systemctl restart picam-yolo
# Cameras are auto-discovered; add --cameras 0 to restrict to one.
PICAM_ARGS=--model models/yolo11n_ncnn_model --imgsz 416 --size 1280x720
EOF
  echo "wrote $DEFAULTS"
else
  echo "keeping existing $DEFAULTS"
fi

cat > "$UNIT" <<EOF
[Unit]
Description=picam-yolo multi-camera YOLO server
Documentation=https://github.com/ubiquitousidea/picam-yolo
After=network-online.target
Wants=network-online.target
# Give up rather than crash-loop forever on a permanent fault (missing model,
# bad arguments) -- otherwise the journal fills and the real error scrolls away.
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
User=$SVC_USER
WorkingDirectory=$REPO_DIR
Environment=PYTHONUNBUFFERED=1
# These, not --threads, are what actually bind: detector.py sets the same
# variables with os.environ.setdefault, so anything exported here wins.
Environment=OMP_NUM_THREADS=$THREADS
Environment=MKL_NUM_THREADS=$THREADS
EnvironmentFile=$DEFAULTS
ExecStart=$VENV_PY -u -m picam_yolo.server \$PICAM_ARGS
Restart=always
RestartSec=5
# Widen only with a supply that negotiates 5V/5A -- see CORES above.
CPUAffinity=$CORES
Nice=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now picam-yolo.service
sleep 3
systemctl --no-pager --lines=0 status picam-yolo.service || true

cat <<EOF

installed. Useful commands:
  sudo systemctl status picam-yolo      # state
  journalctl -u picam-yolo -f           # live log
  sudo systemctl restart picam-yolo     # after editing $DEFAULTS
  sudo systemctl disable --now picam-yolo   # stop and remove from boot
EOF
