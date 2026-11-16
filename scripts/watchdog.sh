#!/bin/bash
# Copyright 2026 the macaq-mlx authors. Licensed under the Apache License, Version 2.0.
#
# Run a command and stop it (and its children) as soon as macOS reports memory pressure level >= 2
# (kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical). Useful on a 16 GB Mac or a shared machine:
# a conversion that outgrows memory is stopped before the system starts swapping hard. macaq-port resumes from its
# per-pass checkpoints when run again.
#
# usage: scripts/watchdog.sh LOG_FILE COMMAND [ARGS...]
# exit status: the command's, or 137 when the watchdog stopped it.
set -uo pipefail
if [ $# -lt 2 ]; then echo "usage: $0 LOG_FILE COMMAND [ARGS...]" >&2; exit 2; fi
log=$1; shift
tree() { local c; for c in $(pgrep -P "$1"); do tree "$c"; done; echo "$1"; }
"$@" > "$log" 2>&1 &
pid=$!
maxlvl=1
stopped=0
while kill -0 $pid 2>/dev/null; do
  lvl=$(sysctl -n kern.memorystatus_vm_pressure_level)
  [ "$lvl" -gt "$maxlvl" ] && maxlvl=$lvl
  if [ "$lvl" -ge 2 ]; then
    echo "-- memory pressure level $lvl, stopping $(date +%H:%M:%S)"
    ps=$(tree $pid); kill $ps 2>/dev/null; sleep 3; kill -9 $ps 2>/dev/null
    stopped=1
    break
  fi
  sleep 1
done
rc=0; wait $pid || rc=$?
[ $stopped -eq 1 ] && rc=137
echo "-- exit $rc, max memory pressure level $maxlvl (log: $log)"
exit $rc
