FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps - use Python 3.12 (LeRobot v0.4.1+ requires it for dataclass compatibility)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git git-lfs curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.4)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install LeRobot v0.4.1 (matching the version in the sim container)
# The async server module is at lerobot.async_inference.policy_server
RUN pip install "lerobot[async]==0.4.1"

# Install SmolVLA dependencies
RUN pip install "lerobot[smolvla]==0.4.1" || true
RUN pip install transformers accelerate safetensors num2words

# Ensure compatible dependency versions
RUN pip install "huggingface-hub>=0.34.2,<0.36.0" \
    "diffusers>=0.35.2" \
    "packaging>=24.2"

WORKDIR /app

# Patch: LeIsaac's client doesn't send rename_map, so default to empty dict
RUN sed -i 's|"rename_map": policy_specs.rename_map|"rename_map": getattr(policy_specs, "rename_map", {})|g' \
    /opt/venv/lib/python3.12/site-packages/lerobot/async_inference/policy_server.py

# Patch: Remap observation image keys (e.g. wrist/front -> camera1/camera2/camera3)
COPY patch_helpers.py /tmp/patch_helpers.py
RUN python /tmp/patch_helpers.py

# Default: start the policy server on 0.0.0.0:8080
EXPOSE 8080

ENTRYPOINT ["python", "-m", "lerobot.async_inference.policy_server"]
CMD ["--host=0.0.0.0", "--port=8080"]