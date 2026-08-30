#!/usr/bin/env bash
# Drive the Pi-side server from this machine.
#
#   ./scripts/piserver.sh start [server args...]   # stop stale, launch detached
#   ./scripts/piserver.sh stop
#   ./scripts/piserver.sh status
#   ./scripts/piserver.sh log [-f]
#
# Launching over SSH is fiddly: a backgrounded remote command can hold the
# channel open and hang your terminal with no output, which looks exactly like
# a failure while the server is in fact running fine. `ssh -f` avoids that by
# backgrounding the client itself once the command is started.
set -euo pipefail

HOST="${HOST:-rpi}"
REMOTE_DIR="${REMOTE_DIR:-picam-yolo}"
LOG="$REMOTE_DIR/run.log"
# The [p] is load-bearing: a plain `pkill -f picam_yolo.server` also matches the
# `bash -c` wrapper carrying the pattern and kills its own SSH session.
PATTERN='[p]icam_yolo.server'
UNIT="${UNIT:-picam-yolo.service}"
# systemctl --user over a non-interactive ssh has no login session to infer the
# user manager's socket from. Single quotes: this expands on the Pi, not here.
UCTL='XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)} systemctl --user'

# True when the systemd unit -- not a hand-started run -- owns the server.
unit_owns_server() { ssh -n "$HOST" "$UCTL is-active --quiet $UNIT" 2>/dev/null; }

cmd="${1:-start}"; shift || true

case "$cmd" in
  stop)
    # Restart=always means a bare pkill is undone five seconds later, so when
    # the unit owns the server, stopping the unit is the only thing that sticks.
    if unit_owns_server; then
      ssh -n "$HOST" "$UCTL stop $UNIT"
      echo "stopped $UNIT (the unit owned this server; see scripts/piservice.sh)"
    else
      ssh -n "$HOST" "pkill -f '$PATTERN' || true"
      echo "stop signalled"
    fi
    ;;

  status)
    ssh -n "$HOST" "pgrep -af '$PATTERN' || echo '(not running)'"
    ;;

  log)
    ssh -n "$HOST" "tail ${1:--n 30} $LOG"
    ;;

  start)
    args=("$@")
    if [[ ${#args[@]} -eq 0 ]]; then
      args=(--backend none --size 1280x720)
    fi

    # Racing the unit for port 5555 is unwinnable: pkill, then systemd
    # restarts it and takes the port back before this launch binds.
    if unit_owns_server; then
      echo "$UNIT is running the server; stop it first:" >&2
      echo "  ./scripts/piservice.sh stop" >&2
      exit 1
    fi

    ssh -n "$HOST" "pkill -f '$PATTERN' || true"
    # Wait for the old process to release the PUB port; binding while it is
    # still held is what produces "Address already in use".
    ssh -n "$HOST" 'for i in $(seq 20); do ss -ltn 2>/dev/null | grep -q ":5555 " || exit 0; sleep 0.25; done; exit 1' \
      || { echo "port 5555 still held after 5s; run '$0 status'" >&2; exit 1; }

    # CORES pins the server to a subset of CPUs and THREADS sizes inference to
    # match. All four are only safe with a supply that negotiates 5V/5A -- on an
    # under-spec one the board browns out and reboots, so CORES=0,1 THREADS=1 is
    # the fallback. CORES=all leaves affinity and thread counts alone entirely.
    cores="${CORES:-0-3}"; threads="${THREADS:-4}"
    prefix=""
    if [[ "$cores" != "all" ]]; then
      prefix="env OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads taskset -c $cores nice -n 5 "
      printf 'pinned to cores %s, %s inference thread(s) (set CORES=all to disable)\n' "$cores" "$threads"
    fi

    printf 'starting: %s\n' "${args[*]}"
    # The backgrounded ssh must not inherit our stdout: -f keeps the client
    # alive for the life of the remote command, so an inherited pipe never sees
    # EOF and any `piserver.sh start | tail` hangs forever with no output.
    ssh -f -n "$HOST" "cd $REMOTE_DIR && setsid ${prefix}.venv/bin/python -u -m picam_yolo.server ${args[*]} > run.log 2>&1 < /dev/null" >/dev/null 2>&1

    # Confirm it actually came up rather than reporting optimistic success.
    # The wait runs entirely on the Pi: polling from here opened a fresh SSH
    # connection per iteration, and the handshakes alone stretched a two-second
    # startup into minutes of apparent hang.
    outcome=$(ssh -n "$HOST" "for i in \$(seq 120); do
        grep -q 'publishing on' '$LOG' 2>/dev/null && { echo UP; exit 0; }
        grep -qE 'ERROR|Traceback' '$LOG' 2>/dev/null && { echo FAIL; exit 0; }
        sleep 0.5
      done; echo TIMEOUT")

    case "$outcome" in
      UP)
        echo "server up:"
        ssh -n "$HOST" "grep -aE 'using [0-9]+ camera|publishing on' '$LOG' | tail -3"
        exit 0 ;;
      FAIL)
        echo "server failed to start:" >&2
        ssh -n "$HOST" "grep -avE 'libcamera|libpisp|IPAProxy|camera_manager|pisp.cpp' '$LOG' | tail -15" >&2
        exit 1 ;;
      *)
        echo "timed out after 60s waiting for startup; check: $0 log" >&2
        exit 1 ;;
    esac
    ;;

  *)
    echo "usage: $0 {start|stop|status|log}" >&2; exit 2 ;;
esac
