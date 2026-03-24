FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Add deadsnakes PPA for Python 3.12
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# System dependencies: GL/X11 for live rendering + MuJoCo deps
RUN apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    curl \
    cmake \
    build-essential \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglfw3 \
    libglfw3-dev \
    libgles2-mesa-dev \
    libosmesa6-dev \
    libglu1-mesa \
    x11-utils \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxxf86vm1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install pip for python3.12 using get-pip.py
RUN apt-get update && apt-get install -y curl \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && rm -rf /var/lib/apt/lists/*

# Install lerobot with libero support
RUN pip install "lerobot[libero, smolvla]"

WORKDIR /workspace

COPY smolvla_libero_eval.sh /smolvla_libero_eval.sh
RUN chmod +x /smolvla_libero_eval.sh
ENTRYPOINT ["/smolvla_libero_eval.sh"]