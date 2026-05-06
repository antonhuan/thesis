# ============================================================
# LeRobot SO-101 — Dockerfile
# Feetech STS3215 motor support + USB serial passthrough
# Python 3.12
# ============================================================

FROM python:3.12-slim

# ── System labels ────────────────────────────────────────────
LABEL maintainer="lerobot-so101"
LABEL description="LeRobot environment for SO-101 robot arm (Feetech motors)"

# ── Prevent interactive prompts ──────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Build dependencies + ffmpeg libs (required by LeRobot) ───
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    cmake \
    build-essential \
    python3-dev \
    pkg-config \
    # ffmpeg + PyAV build deps
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    # USB/serial for Feetech bus
    udev \
    usbutils \
    libusb-1.0-0 \
    # OpenCV headless deps
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    # Misc
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Upgrade pip & base tools ─────────────────────────────────
RUN pip install --upgrade pip setuptools wheel

# ── Clone LeRobot ────────────────────────────────────────────
WORKDIR /opt
RUN git clone https://github.com/huggingface/lerobot.git

# ── Install LeRobot with Feetech extras ──────────────────────
# --no-binary=av forces PyAV to build against the system ffmpeg
# libs installed above, avoiding codec/encoder mismatches.
WORKDIR /opt/lerobot
RUN pip install --no-binary=av -e ".[feetech]"

# ── Working directory for user scripts / calibration data ────
WORKDIR /workspace

# ── Runtime environment ──────────────────────────────────────
# HF_HOME: cache dir for downloaded models & datasets
ENV HF_HOME=/workspace/.cache/huggingface

# ── Expose calibration & dataset volumes ─────────────────────
VOLUME ["/workspace", "/workspace/.cache"]

# ── Entrypoint ───────────────────────────────────────────────
# Drops into bash by default; override CMD for specific tasks.
#
# Examples:
#   Find USB port:
#     docker run --rm --device=/dev/ttyUSB0 lerobot-so101 lerobot-find-port
#
#   Setup motors (follower):
#     docker run --rm --device=/dev/ttyUSB0 lerobot-so101 \
#       lerobot-setup-motors \
#         --robot.type=so101_follower \
#         --robot.port=/dev/ttyUSB0
#
#   Calibrate (follower):
#     docker run --rm --device=/dev/ttyUSB0 -v $(pwd)/data:/workspace lerobot-so101 \
#       lerobot-calibrate \
#         --robot.type=so101_follower \
#         --robot.port=/dev/ttyUSB0 \
#         --robot.id=my_follower_arm
#
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["bash"]