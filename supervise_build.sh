#!/bin/bash
# Keep the staging build alive without a human watching it.
#
# Two failure modes have actually happened here, and liveness alone catches neither:
#   * the process exits (missing binary, bad archive)
#   * the process lives but makes no progress (Hub throttling wedged it for 12 minutes
#     with 0 MB/s and the CPU 99% idle)
# So this watches log *age*, not just the pid, and relaunches on either. Per-source
# manifests make a relaunch cheap: finished sources are reused, not re-downloaded.
set -u

LOG=/root/build_all.log
STATE=/root/supervisor.log
STALL_SECONDS="${STALL_SECONDS:-900}"
: "${HF_TOKEN:?set HF_TOKEN in the environment}"

note() { echo "$(date -u +%FT%TZ) $*" >> "$STATE"; }

launch() {
    note "launching build"
    mv -f "$LOG" "/root/build_all.$(date -u +%H%M%S).log" 2>/dev/null || true
    setsid nohup /root/nemo-he/build_all.sh > "$LOG" 2>&1 < /dev/null &
    sleep 60
}

note "supervisor started (stall threshold ${STALL_SECONDS}s)"

while true; do
    # Finished cleanly? train.json is only written after every source is handled.
    if grep -q "wrote /root/data/manifests/train.json" "$LOG" 2>/dev/null; then
        note "BUILD COMPLETE"
        exit 0
    fi

    if ! pgrep -f 'build_hf_manifests[.]py' > /dev/null 2>&1; then
        note "process gone -- relaunching"
        launch
        continue
    fi

    if [ -f "$LOG" ]; then
        age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
        if [ "$age" -gt "$STALL_SECONDS" ]; then
            note "STALLED ${age}s with no log output -- killing and relaunching"
            pkill -9 -f 'build_hf_manifests[.]py' 2>/dev/null || true
            sleep 5
            launch
            continue
        fi
    fi

    sleep 60
done
