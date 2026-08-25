#!/usr/bin/env bash
# Start and stop the picam-yolo *service* on the Pi from this machine.
#
#   ./scripts/piservice.sh stop        # stop now; systemd starts it again at boot
#   ./scripts/piservice.sh start
#   ./scripts/piservice.sh restart
#   ./scripts/piservice.sh status
#   ./scripts/piservice.sh off         # stop *and* disable: stays off across reboots
#   ./scripts/piservice.sh on          # enable and start
#
# Stopping the unit is the lever that actually drops the Pi's power draw: it is
# the camera capture and the NCNN inference that keep two cores busy, and both
# go away with the process.
#
# Use this rather than `piserver.sh stop` whenever the unit is installed. That
# one kills the process, and Restart=always brings it back five seconds later,
# which reads as "stop did nothing".
set -euo pipefail

HOST="${HOST:-rpi}"
UNIT="${UNIT:-picam-yolo.service}"
REMOTE_DIR="${REMOTE_DIR:-picam-yolo}"
LOG="$REMOTE_DIR/run.log"
# The [p] is load-bearing: a plain `pkill -f picam_yolo.server` also matches the
# `bash -c` wrapper carrying the pattern and kills its own SSH session.
PATTERN='[p]icam_yolo.server'

# A non-interactive ssh has no login session, so systemctl --user cannot find
# the user manager without being told where its socket lives. Single quotes:
# this expands on the Pi, not here.
UCTL='XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)} systemctl --user'

cmd="${1:-status}"

# Which flavour of unit is installed, in one round trip rather than three.
# `systemctl cat` is read-only and needs no privileges either way.
mode=$(ssh -n "$HOST" "
  if $UCTL cat $UNIT >/dev/null 2>&1; then echo user
  elif systemctl cat $UNIT >/dev/null 2>&1; then echo system
  else echo none; fi")

sc() { ssh -n "$HOST" "$UCTL $*"; }

# The unit *appends* to run.log (piserver.sh start truncates it), so a stale
# "publishing on" from an earlier run would otherwise confirm a start that in
# fact failed. Record the length first and read only what this start adds.
log_baseline() { ssh -n "$HOST" "wc -l < '$LOG' 2>/dev/null || echo 0"; }

# Report what actually happened rather than assuming the command took. The wait
# runs entirely on the Pi: polling from here costs an SSH handshake per attempt.
confirm_up() {
  base="${1:-0}"
  outcome=$(ssh -n "$HOST" "for i in \$(seq 40); do
      $UCTL is-active --quiet $UNIT || { echo DEAD; exit 0; }
      tail -n +$((base + 1)) '$LOG' 2>/dev/null | grep -q 'publishing on' && { echo UP; exit 0; }
      sleep 0.5
    done; echo TIMEOUT")
  case "$outcome" in
    UP)   echo "running:"
          ssh -n "$HOST" "tail -n +$((base + 1)) '$LOG' | grep -aE 'using [0-9]+ camera|publishing on' | tail -2" ;;
    DEAD) echo "unit exited; last log lines:" >&2
          ssh -n "$HOST" "tail -n +$((base + 1)) '$LOG' | grep -avE 'libcamera|libpisp|IPAProxy|camera_manager|pisp.cpp' | tail -15" >&2
          exit 1 ;;
    *)    echo "active, but no 'publishing on' after 20s; check: $0 status" >&2; exit 1 ;;
  esac
}

# Everything except status mutates the unit, and only the user unit can be
# driven from here.
case "$cmd" in
  stop|start|restart|on|off)
    case "$mode" in
      user) ;;
      system)
        # sudo on this Pi wants a password and a TTY, and `ssh -t` cannot
        # supply one when stdin is not a terminal either.
        echo "$UNIT is a *system* unit, which this script cannot drive: sudo here" >&2
        echo "needs a password and a TTY. Run it on the Pi itself:" >&2
        echo >&2
        echo "  ssh $HOST" >&2
        case "$cmd" in
          off) verb="disable --now" ;;
          on)  verb="enable --now" ;;
          *)   verb="$cmd" ;;
        esac
        echo "  sudo systemctl $verb $UNIT" >&2
        exit 1 ;;
      none)
        if [[ "$cmd" == "stop" || "$cmd" == "off" ]]; then
          # No unit, but a hand-started server may still be burning cores.
          ssh -n "$HOST" "pkill -f '$PATTERN' || true"
          echo "no $UNIT installed; killed any hand-started server instead"
          exit 0
        fi
        echo "no $UNIT installed on $HOST." >&2
        echo "install it:  ssh $HOST 'bash ~/$REMOTE_DIR/scripts/install_user_service.sh'" >&2
        echo "or run ad hoc:  ./scripts/piserver.sh start" >&2
        exit 1 ;;
    esac
    ;;
esac

case "$cmd" in
  stop)
    sc stop "$UNIT"
    echo "stopped. Still enabled, so it returns at the next boot -- use '$0 off' to prevent that."
    ;;

  off)
    sc disable --now "$UNIT"
    echo "stopped and disabled; stays off across reboots. Bring it back with '$0 on'."
    ;;

  start)
    base=$(log_baseline)
    sc start "$UNIT"
    confirm_up "$base"
    ;;

  on)
    base=$(log_baseline)
    sc enable --now "$UNIT"
    confirm_up "$base"
    ;;

  restart)
    base=$(log_baseline)
    sc restart "$UNIT"
    confirm_up "$base"
    ;;

  status)
    case "$mode" in
      user)
        ssh -n "$HOST" "
          printf 'unit    : %s (user)\n' '$UNIT'
          printf 'active  : %s\n' \"\$($UCTL is-active $UNIT || true)\"
          printf 'enabled : %s\n' \"\$($UCTL is-enabled $UNIT || true)\"
          printf 'process : %s\n' \"\$(pgrep -af '$PATTERN' || echo '(none)')\"" ;;
      system)
        ssh -n "$HOST" "systemctl --no-pager --lines=0 status $UNIT || true" ;;
      none)
        echo "unit    : (not installed)"
        ssh -n "$HOST" "printf 'process : %s\n' \"\$(pgrep -af '$PATTERN' || echo '(none)')\"" ;;
    esac
    ;;

  *)
    echo "usage: $0 {stop|start|restart|status|off|on}" >&2; exit 2 ;;
esac
