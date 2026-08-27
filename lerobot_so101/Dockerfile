# ============================================================
# LeRobot SO-101 — Dockerfile
# Feetech STS3215 motor support + USB serial passthrough
# NVIDIA CUDA base for GPU-accelerated Rerun viewer
# ============================================================
FROM nvidia/cuda:12.6.0-base-ubuntu24.04

# ── System labels ────────────────────────────────────────────
LABEL maintainer="lerobot-so101"
LABEL description="LeRobot environment for SO-101 robot arm (Feetech motors)"

# ── Prevent interactive prompts ──────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Python 3.12 + build dependencies + ffmpeg libs ───────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    curl \
    wget \
    cmake \
    build-essential \
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
    # OpenCV deps
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    # Rerun viewer X11/Vulkan deps
    libgtk-3-dev \
    libxkbcommon-x11-0 \
    vulkan-tools \
    # Speech (for lerobot play_sounds)
    speech-dispatcher \
    # Misc
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Make python3.12 the default python ───────────────────────
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# ── Upgrade pip & base tools ────────────────────────────────

# ── Clone LeRobot ────────────────────────────────────────────
WORKDIR /opt
RUN git clone https://github.com/huggingface/lerobot.git

# ── Install LeRobot with Feetech extras ──────────────────────
# --no-binary=av forces PyAV to build against the system ffmpeg
# libs installed above, avoiding codec/encoder mismatches.
WORKDIR /opt/lerobot
RUN pip install --no-binary=av -e ".[feetech]" --break-system-packages --ignore-installed packaging

# ── Fix OpenCV: replace headless with full build for V4L2 camera support
RUN pip install --force-reinstall opencv-python --break-system-packages && \
    pip install "numpy<2.3.0,>=2.0.0" --break-system-packages

# ── Rerun + viz deps ────────────────────────────────────────
RUN pip install rerun-sdk pynput --break-system-packages

RUN pip install lerobot[dataset] --break-system-packages

RUN pip install pyrealsense2 --break-system-packages

RUN pip install lerobot[pi] --break-system-packages

RUN pip install num2words --break-system-packages

RUN pip install openpi-client --break-system-packages

RUN pip install lerobot[async] --break-system-packages

RUN pip install lerobot[training] --break-system-packages

RUN pip install vllm qwen-vl-utils pillow --break-system-packages

RUN pip install placo --break-system-packages

# ── Non-root runtime user ─────────────────────────────────────
# ./data is bind-mounted (not a named volume), so whatever UID/GID writes
# inside the container is exactly what ends up owning the files on the host.
# Without this the default root user leaves root-owned logs/outputs/etc.
# behind in the repo. Defaults match the host dev user; override at build
# time (--build-arg USER_UID=... USER_GID=...) if yours differs.
# dialout/video/plugdev are added so the arm's serial port and cameras under
# /dev (root:dialout / root:video, mode 660) stay reachable without root.
ARG USER_UID=1002
ARG USER_GID=1002
RUN groupadd -f -g 20 dialout && \
    groupadd -f -g 44 video && \
    groupadd -f -g 46 plugdev && \
    groupadd -g ${USER_GID} appuser && \
    useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash appuser && \
    usermod -aG dialout,video,plugdev appuser

USER appuser

# ── Working directory for user scripts / calibration data ────
WORKDIR /workspace

# ── Runtime environment ──────────────────────────────────────
ENV HF_HOME=/workspace/.cache/huggingface

# ── Expose calibration & dataset volumes ─────────────────────
VOLUME ["/workspace", "/workspace/.cache"]

# ── Entrypoint ───────────────────────────────────────────────
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["bash"]