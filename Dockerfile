FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies: GL/X11 for live rendering + MuJoCo deps
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    # X11 / OpenGL for live sim window
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglfw3 \
    libglfw3-dev \
    libgles2-mesa-dev \
    libosmesa6-dev \
    libglu1-mesa \
    x11-utils \
    # MuJoCo / robosuite runtime deps
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxxf86vm1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install lerobot with libero support
RUN pip install "lerobot[libero] @ git+https://github.com/huggingface/lerobot.git"

WORKDIR /workspace

# Entrypoint: run eval directly
ENTRYPOINT ["lerobot-eval"]
