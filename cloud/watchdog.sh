#!/bin/bash
# HackGuard auto-shutdown watchdog. Installed on the EC2 machine at first boot and run every
# 5 minutes by a systemd timer (see cloud/user_data.sh).
#
# The machine counts as BUSY if any of these is true:
#   - the GPU is more than 10% busy,
#   - the pipeline, a test run, a package install or a Docker build is running,
#   - someone typed in an SSH terminal in the last IDLE_MINUTES minutes.
# If it has not been busy for IDLE_MINUTES, the machine shuts itself down. The instance is
# launched with "shutdown behavior = stop", so shutting down STOPS it (no more GPU charges;
# the disk and its data are kept) rather than deleting it.
#
# Hard limit: after MAX_HOURS since boot it shuts down no matter what, as a final safety net.

IDLE_MINUTES=30
MAX_HOURS=36
STATE_DIR=/var/lib/hackguard
STATE="$STATE_DIR/last_busy"

mkdir -p "$STATE_DIR"
now=$(date +%s)
[ -f "$STATE" ] || echo "$now" > "$STATE"

boot=$(date -d "$(uptime -s)" +%s)
if (( now - boot > MAX_HOURS * 3600 )); then
    logger "hackguard-watchdog: $MAX_HOURS hour limit reached, shutting down"
    shutdown -h now "HackGuard: maximum runtime reached"
    exit 0
fi

busy=0
gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ "$gpu" =~ ^[0-9]+$ ]] && (( gpu > 10 )); then busy=1; fi
if pgrep -f "hackguard.pipeline|pytest|pip install|uv pip|docker build|apt-get" > /dev/null; then busy=1; fi
if [ -n "$(find /dev/pts -name '[0-9]*' -mmin -"$IDLE_MINUTES" 2>/dev/null)" ]; then busy=1; fi

if (( busy )); then
    echo "$now" > "$STATE"
    exit 0
fi

last=$(cat "$STATE")
if (( now - last > IDLE_MINUTES * 60 )); then
    logger "hackguard-watchdog: idle for $IDLE_MINUTES minutes, shutting down"
    shutdown -h now "HackGuard: idle for $IDLE_MINUTES minutes"
fi
