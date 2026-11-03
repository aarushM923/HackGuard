# The sandbox that runs model-written code on the EC2 machine.
# Build with: docker build -t hackguard-sandbox -f docker/sandbox.Dockerfile docker/
#
# It contains only Python, pytest and numpy (the EvalPlus comparison helpers need numpy).
# Each test run starts a fresh container from this image with no network access and
# limited memory, CPU and processes (see DockerRunner in hackguard/tasks/workspace.py).
FROM python:3.11-slim

RUN pip install --no-cache-dir pytest==9.2.0 numpy==2.4.6

WORKDIR /work
