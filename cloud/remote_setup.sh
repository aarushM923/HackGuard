#!/bin/bash
# One-time setup on the EC2 machine, run from ~/HackGuard after `ec2 push`.
# Creates the Python environment, builds the Docker sandbox, and runs the tests (including
# the Docker sandbox test that is skipped on the laptop) before any GPU money is spent on
# real runs.
set -euo pipefail
cd ~/HackGuard

echo "== Waiting for first-boot setup to finish =="
while [ ! -f /var/lib/hackguard-ready ]; do sleep 5; done

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "== Python environment =="
python3.11 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements-gpu.txt

echo "== Docker sandbox image =="
# `sg docker` runs the command with the docker group active, even in this first login session.
sg docker -c "docker build -q -t hackguard-sandbox -f docker/sandbox.Dockerfile docker/"

echo "== Tests =="
sg docker -c ".venv/bin/python -m pytest -q"

echo "Setup finished."
