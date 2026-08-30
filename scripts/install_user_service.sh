#!/usr/bin/env bash
# Install picam-yolo as a *user* systemd service. Needs no root, so it works on
# a Pi where sudo requires a password and a TTY.
#
#   ssh rpi 'bash ~/picam-yolo/scripts/install_user_service.sh'
#
# Prefer scripts/install_service.sh (system unit) where passwordless sudo exists.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/picam-yolo}"
VENV_PY="$REPO_DIR/.venv/bin/python"
# Cores the server may use, and the inference threads to match. Four browns out
# a board fed by an under-spec supply, so check before widening:
#   od -An -tu4 --endian=big /sys/firmware/devicetree/base/chosen/power/max_current
# 5000 means the supply negotiated 5V/5A and four cores are affordable; 3000 is
# the conservative fallback, and CORES=0,1 THREADS=1 is the safe setting there.
CORES="${CORES:-0-3}"
THREADS="${THREADS:-4}"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/picam-yolo.service"
ENVFILE="$HOME/.config/picam-yolo.env"

# A non-interactive ssh has no login session, so systemctl --user cannot find
# the user manager without being told where its socket lives.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

[[ -x "$VENV_PY" ]] || { echo "no venv at $VENV_PY -- run scripts/setup_pi.sh first" >&2; exit 1; }

# Without linger, user units stop at logout and never start at boot.
loginctl enable-linger "$USER"

mkdir -p "$UNIT_DIR" "$(dirname "$ENVFILE")"
if [[ ! -f "$ENVFILE" ]]; then
  cat > "$ENVFILE" <<EOF
# Arguments for picam-yolo. Edit, then: systemctl --user restart picam-yolo
# Cameras are auto-discovered; add --cameras 0 to restrict to one.
PICAM_ARGS=--model models/yolo11n_ncnn_model --imgsz 416 --size 1280x720
EOF
  echo "wrote $ENVFILE"
else
  echo "keeping existing $ENVFILE"
fi

# A hand-started server would hold the PUB port and make the unit's first start
# fail for a reason that looks nothing like the real cause.
pkill -f '[p]icam_yolo.server' 2>/dev/null || true

cat > "$UNIT" <<EOF
[Unit]
Description=picam-yolo multi-camera YOLO server
Documentation=https://github.com/ubiquitousidea/picam-yolo
# Cameras may not be enumerated yet this early in boot; Restart handles that,
# and StartLimit stops a permanent fault from crash-looping forever.
StartLimitIntervalSec=300
StartLimitBurst=10

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
Environment=PYTHONUNBUFFERED=1
# These, not --threads, are what actually bind: detector.py sets the same
# variables with os.environ.setdefault, so anything exported here wins.
Environment=OMP_NUM_THREADS=$THREADS
Environment=MKL_NUM_THREADS=$THREADS
EnvironmentFile=$ENVFILE
# taskset rather than the CPUAffinity= directive: narrowing your own affinity
# needs no privileges, while the unit directive can be refused in a user
# manager.
ExecStart=/usr/bin/taskset -c $CORES $VENV_PY -u -m picam_yolo.server \$PICAM_ARGS
Restart=always
RestartSec=5
Nice=5
# This Pi keeps no user journal ("No journal files were found"), so unit output
# would otherwise vanish. Append to the same run.log that scripts/piserver.sh
# already tails. It grows unbounded; truncate it if it ever matters.
StandardOutput=append:$REPO_DIR/run.log
StandardError=append:$REPO_DIR/run.log

[Install]
WantedBy=default.target
EOF

echo "pinning to cores $CORES with $THREADS inference thread(s)"
systemctl --user daemon-reload
systemctl --user enable --now picam-yolo.service
sleep 4
systemctl --user --no-pager --lines=0 status picam-yolo.service || true

cat <<EOF

installed as a user service (starts at boot; linger is on).
  systemctl --user status picam-yolo
  tail -f $REPO_DIR/run.log
  systemctl --user restart picam-yolo      # after editing $ENVFILE
  systemctl --user disable --now picam-yolo
EOF
