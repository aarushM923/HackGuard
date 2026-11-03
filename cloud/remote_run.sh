#!/bin/bash
# Start a pipeline run on the EC2 machine inside tmux, so it keeps running after you
# disconnect. Usage: cloud/remote_run.sh <preset> [extra pipeline arguments]
#   e.g. cloud/remote_run.sh pilot
#        cloud/remote_run.sh gpu
# Watch it with:   tmux attach -t hackguard    (detach again with Ctrl-b then d)
# Log file:        ~/HackGuard/runs/<preset>/log.txt
# When the run ends the GPU goes idle, and the watchdog stops the machine 30 minutes later.
set -euo pipefail
cd ~/HackGuard
PRESET="$1"; shift || true
RUN="runs/$PRESET"
mkdir -p "$RUN"
[ -f "$RUN/config.json" ] || .venv/bin/python -m hackguard.pipeline init --preset "$PRESET"
tmux new-session -d -s hackguard \
  "sg docker -c '.venv/bin/python -m hackguard.pipeline all --run $RUN $*' 2>&1 | tee -a $RUN/log.txt"
echo "Started. Attach with: tmux attach -t hackguard"
