#!/bin/bash
# Runs once, as root, when the EC2 machine first boots ("user data").
# 1. Installs the auto-shutdown watchdog (cloud/watchdog.sh is pasted in below by ec2.py).
# 2. Installs Python 3.11 (Ubuntu 22.04 ships 3.10) and rsync.
# 3. Lets the ubuntu user run Docker without sudo.
set -euxo pipefail

cat > /usr/local/bin/hackguard-watchdog.sh <<'WATCHDOG'
__WATCHDOG_SCRIPT__
WATCHDOG
chmod +x /usr/local/bin/hackguard-watchdog.sh

cat > /etc/systemd/system/hackguard-watchdog.service <<'UNIT'
[Unit]
Description=HackGuard idle shutdown check
[Service]
Type=oneshot
ExecStart=/usr/local/bin/hackguard-watchdog.sh
UNIT

cat > /etc/systemd/system/hackguard-watchdog.timer <<'UNIT'
[Unit]
Description=Run the HackGuard idle shutdown check every 5 minutes
[Timer]
OnBootSec=5min
OnUnitActiveSec=5min
[Install]
WantedBy=timers.target
UNIT

systemctl daemon-reload
systemctl enable --now hackguard-watchdog.timer

export DEBIAN_FRONTEND=noninteractive
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.11 python3.11-venv python3.11-dev rsync tmux
usermod -aG docker ubuntu
touch /var/lib/hackguard-ready
