#!/bin/bash
# Keep training alive unattended.
#
# Watches for two distinct failures, because liveness alone catches neither:
#   * the process exits (OOM, NCCL teardown, disk full)
#   * the process lives but stops progressing (dataloader wedge, silent hang)
# Relaunch is safe: run_training.sh initialises from the best .nemo, so a restart
# resumes from the best weights rather than from the base model.
#
# It deliberately does NOT restart after a clean early-stop -- that is the signal we
# want to wake up to, not something to paper over.
set -u

LOG=${LOG:-/root/train5.log}
STATE=/root/train_supervisor.log
STALL_SECONDS=${STALL_SECONDS:-1800}
MAX_RESTARTS=${MAX_RESTARTS:-10}
restarts=0

note() { echo "$(date -u +%FT%TZ) $*" >> "$STATE"; }

note "supervisor started (log=$LOG stall=${STALL_SECONDS}s)"

while true; do
    if grep -aq "Monitored metric val_wer did not improve" "$LOG" 2>/dev/null; then
        note "EARLY STOPPING fired -- training converged, not restarting"
        exit 0
    fi

    if ! pgrep -f "finetune[.]py" > /dev/null 2>&1; then
        if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
            note "process gone but restart budget ($MAX_RESTARTS) exhausted -- giving up"
            exit 1
        fi
        restarts=$((restarts + 1))
        note "process gone -- relaunch #$restarts"
        mv -f "$LOG" "${LOG%.log}.$(date -u +%H%M%S).log" 2>/dev/null || true
        setsid nohup env HF_TOKEN=x bash /root/nemo-he/run_training.sh > "$LOG" 2>&1 < /dev/null &
        sleep 300   # model load + manifest parse takes minutes; don't re-trip immediately
        continue
    fi

    if [ -f "$LOG" ]; then
        age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
        if [ "$age" -gt "$STALL_SECONDS" ]; then
            note "STALLED ${age}s with no output -- killing for relaunch"
            pkill -9 -f "finetune[.]py" 2>/dev/null || true
            pkill -9 -f "nemo_finetune_entr[y]" 2>/dev/null || true
            sleep 10
            nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
            sleep 5
            continue
        fi
    fi

    sleep 120
done
