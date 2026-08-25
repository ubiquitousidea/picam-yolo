#!/usr/bin/env bash
# Push the working tree to the Pi. Code only -- models are synced separately
# because they are large and change rarely.
set -euo pipefail

HOST="${HOST:-rpi}"
REMOTE_DIR="${REMOTE_DIR:-picam-yolo}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# run.log is Pi-side state and exists nowhere else: --delete would unlink it
# while the server still holds it open, so output would vanish into the
# unlinked inode until the next restart. recordings/ is client-side output
# (gitignored here) and has no business on the Pi -- without this it gets
# pushed there, and --delete then mirrors local deletions back.
rsync -az --delete \
  --exclude '.venv/' --exclude '__pycache__/' --exclude '.git/' \
  --exclude 'models/' --exclude '*.egg-info/' \
  --exclude 'run.log' --exclude 'recordings/' \
  "$LOCAL_DIR/" "$HOST:$REMOTE_DIR/"

echo "synced $LOCAL_DIR -> $HOST:$REMOTE_DIR"

if [[ "${WITH_MODELS:-0}" == "1" ]]; then
  rsync -az "$LOCAL_DIR/models/" "$HOST:$REMOTE_DIR/models/"
  echo "synced models/"
fi
