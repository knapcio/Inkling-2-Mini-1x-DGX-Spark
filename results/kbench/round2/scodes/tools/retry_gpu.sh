#!/bin/bash
# Re-queue a kbrun job until it actually runs (kbrun gives up after 60 min of waiting with exit 3).
# usage: retry_gpu.sh LOG TIMEOUT_MIN "CMD"
LOG=$1; TMO=$2; CMD=$3
for i in $(seq 1 8); do
  ssh -n box-b "~/inkling/r2/kbrun.sh scodes $TMO \"$CMD\"" > "$LOG.try" 2>&1
  if grep -q "KBRUN start" "$LOG.try"; then mv "$LOG.try" "$LOG"; exit 0; fi
  cat "$LOG.try" >> "$LOG.waits"; sleep 30
done
exit 3
