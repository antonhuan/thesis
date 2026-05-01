FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-6 \
    libxi6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libvulkan1 \
    vulkan-tools \
    libxt6 \
    libglu1-mesa \
    libxmu6 \
    libxpm4 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment with python3.11
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install setuptools (pin to version that includes pkg_resources)
RUN pip install --upgrade pip "setuptools<81" wheel

# Step 1: Install PyTorch with CUDA 12.8
RUN pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Step 2: Pre-install flatdict (its build fails in pip's isolated env)
RUN pip install --no-build-isolation flatdict

# Step 3: Clone LeIsaac repo and install in editable mode with IsaacLab/IsaacSim extras
ENV SETUPTOOLS_USE_DISTUTILS=stdlib
WORKDIR /workspace
RUN git clone --recursive https://github.com/LightwheelAI/leisaac.git
RUN pip install --no-build-isolation -e '/workspace/leisaac/source/leisaac[isaaclab]' \
    --extra-index-url https://pypi.nvidia.com

# Step 4: Install LeRobot
RUN pip install lerobot==0.4.1

# Step 5: Install async gRPC deps for the policy client
RUN pip install grpcio protobuf msgpack pydantic

# Step 6: Pin numpy and packaging to Isaac Sim compatible versions
RUN pip install numpy==1.26.0 packaging==23.0

# Step 7: Patch the hardcoded import path in LeIsaac
# LeIsaac hardcodes lerobot.scripts.server.helpers but v0.4.1 moved it to lerobot.async_inference.helpers
RUN sed -i 's|lerobot.scripts.server.helpers|lerobot.async_inference.helpers|g' \
    /workspace/leisaac/source/leisaac/leisaac/policy/lerobot/__init__.py

# No camera modifications — uses stock front + wrist config
# matching the LightwheelAI/leisaac-pick-orange dataset

WORKDIR /workspace/leisaac

CMD ["/bin/bash"]