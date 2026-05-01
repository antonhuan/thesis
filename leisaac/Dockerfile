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

# Step 4: Install LeRobot (without async extra to avoid numpy/packaging conflicts)
RUN pip install lerobot==0.4.1

# Step 5: Install only the async gRPC deps that LeRobot needs for the policy client
# This avoids pulling in numpy>=2 and packaging>=24 which break Isaac Sim
RUN pip install grpcio protobuf msgpack pydantic

# Step 6: Pin numpy and packaging back to Isaac Sim compatible versions
RUN pip install numpy==1.26.0 packaging==23.0

# Step 7: Patch the hardcoded import path in LeIsaac
# LeIsaac hardcodes lerobot.scripts.server.helpers but v0.4.1 moved it to lerobot.async_inference.helpers
RUN sed -i 's|lerobot.scripts.server.helpers|lerobot.async_inference.helpers|g' \
    /workspace/leisaac/source/leisaac/leisaac/policy/lerobot/__init__.py

# Step 8: Add side camera, rename all cameras, and fix definition order
# SmolVLA expects: camera1=top-down, camera2=wrist, camera3=side
# The policy server maps alphabetically, so we name: a_top, b_wrist, c_side
# We also reorder definitions so a_top appears before b_wrist (Python dict insertion order matters)
COPY <<'PATCH_CAMERAS' /tmp/patch_cameras.py
import re
import glob

TASKS_DIR = '/workspace/leisaac/source/leisaac/leisaac/tasks'

# ============================================================
# 1. Patch single_arm_env_cfg.py (parent scene config)
# ============================================================
p = f'{TASKS_DIR}/template/single_arm_env_cfg.py'
txt = open(p).read()

# Add side camera before light definition (if not already present)
if 'side: TiledCameraCfg' not in txt:
    side_block = '''
    side: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/side_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.72684, -0.22668, 0.14343), rot=(-0.5, 0.5, 0.5, -0.5), convention="ros"
        ),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

'''
    txt = txt.replace('    light = AssetBaseCfg(', side_block + '    light = AssetBaseCfg(')

# Add side ObsTerm (if not already present)
if 'SceneEntityCfg("side")' not in txt:
    side_obs = '\n        side = ObsTerm(\n            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("side"), "data_type": "rgb", "normalize": False}\n        )'
    front_obs_end = 'func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front"), "data_type": "rgb", "normalize": False}\n        )'
    txt = txt.replace(front_obs_end, front_obs_end + side_obs)

open(p, 'w').write(txt)
print(f'  Added side camera and ObsTerm to {p}')

# ============================================================
# 2. Patch lift_cube_env_cfg.py — comment out delete_attribute
# ============================================================
p2 = f'{TASKS_DIR}/lift_cube/lift_cube_env_cfg.py'
txt2 = open(p2).read()
txt2 = txt2.replace('delete_attribute(self, "wrist")', '# delete_attribute(self, "wrist")')
txt2 = txt2.replace('delete_attribute(self.policy, "wrist")', '# delete_attribute(self.policy, "wrist")')
open(p2, 'w').write(txt2)
print(f'  Commented out delete_attribute in {p2}')

# ============================================================
# 3. Rename cameras across ALL task files
#    front -> a_top, wrist -> b_wrist, side -> c_side
# ============================================================
rename_files = glob.glob(f'{TASKS_DIR}/**/*.py', recursive=True)

for fpath in rename_files:
    txt = open(fpath).read()
    original = txt

    # Attribute definitions (4-space indent = scene cfg, 8-space = obs terms)
    txt = re.sub(r'^(    )front: TiledCameraCfg', r'\1a_top: TiledCameraCfg', txt, flags=re.MULTILINE)
    txt = re.sub(r'^(    )wrist: TiledCameraCfg', r'\1b_wrist: TiledCameraCfg', txt, flags=re.MULTILINE)
    txt = re.sub(r'^(    )side: TiledCameraCfg', r'\1c_side: TiledCameraCfg', txt, flags=re.MULTILINE)
    txt = re.sub(r'^(        )front = ObsTerm', r'\1a_top = ObsTerm', txt, flags=re.MULTILINE)
    txt = re.sub(r'^(        )wrist = ObsTerm', r'\1b_wrist = ObsTerm', txt, flags=re.MULTILINE)
    txt = re.sub(r'^(        )side = ObsTerm', r'\1c_side = ObsTerm', txt, flags=re.MULTILINE)

    # SceneEntityCfg string references
    txt = txt.replace('SceneEntityCfg("front")', 'SceneEntityCfg("a_top")')
    txt = txt.replace('SceneEntityCfg("wrist")', 'SceneEntityCfg("b_wrist")')
    txt = txt.replace('SceneEntityCfg("side")', 'SceneEntityCfg("c_side")')

    # Bare string references (randomize_camera_uniform, camera lists, overlay paths, etc.)
    # Only match quoted strings that are standalone camera names, not inside prim_path strings
    # We use word boundary matching within quotes
    txt = re.sub(r'"front"(?!_camera)', '"a_top"', txt)
    txt = re.sub(r'"wrist"(?!_camera)', '"b_wrist"', txt)
    txt = re.sub(r'"side"(?!_camera)', '"c_side"', txt)

    # Camera list in single_arm_env.py: for cam in ["front", "wrist", "side"]
    txt = txt.replace('["front", "wrist", "side"]', '["a_top", "b_wrist", "c_side"]')
    # Catch already-partially-renamed variants
    txt = txt.replace('["a_top", "wrist", "side"]', '["a_top", "b_wrist", "c_side"]')

    if txt != original:
        open(fpath, 'w').write(txt)
        print(f'  Renamed cameras in {fpath}')

# ============================================================
# 4. Fix definition order in single_arm_env_cfg.py
#    a_top must appear before b_wrist for correct dict order
# ============================================================
p = f'{TASKS_DIR}/template/single_arm_env_cfg.py'
txt = open(p).read()

b_match = re.search(r'(    b_wrist: TiledCameraCfg = TiledCameraCfg\(.*?\n    \))', txt, re.DOTALL)
a_match = re.search(r'(    a_top: TiledCameraCfg = TiledCameraCfg\(.*?\n    \))', txt, re.DOTALL)

if b_match and a_match:
    b_block = b_match.group(1)
    a_block = a_match.group(1)
    # Only swap if b_wrist appears before a_top
    if txt.index(b_block) < txt.index(a_block):
        txt = txt.replace(b_block, '<<<PLACEHOLDER_A>>>')
        txt = txt.replace(a_block, '<<<PLACEHOLDER_B>>>')
        txt = txt.replace('<<<PLACEHOLDER_A>>>', a_block)
        txt = txt.replace('<<<PLACEHOLDER_B>>>', b_block)
        open(p, 'w').write(txt)
        print(f'  Swapped a_top/b_wrist order in {p}')

# ============================================================
# 5. Remove camera overrides from lift_cube_env_cfg.py
#    so it inherits cleanly from parent
# ============================================================
p2 = f'{TASKS_DIR}/lift_cube/lift_cube_env_cfg.py'
txt2 = open(p2).read()
changed = False
for cam in ['a_top', 'b_wrist', 'c_side']:
    pattern = rf'\n    {cam}: TiledCameraCfg = TiledCameraCfg\(.*?\n    \)\n'
    new_txt = re.sub(pattern, '\n', txt2, flags=re.DOTALL)
    if new_txt != txt2:
        txt2 = new_txt
        changed = True
if changed:
    open(p2, 'w').write(txt2)
    print(f'  Removed camera overrides from {p2}')

print('\nAll camera patches applied successfully.')
PATCH_CAMERAS
RUN python3 /tmp/patch_cameras.py && rm -f /tmp/patch_cameras.py

WORKDIR /workspace/leisaac

# Default command
CMD ["/bin/bash"]