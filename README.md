# LeRobot Sim Eval — Local GPU + Live Display

Runs `lerobot-eval` (pi0.5 on libero) in Docker with your local NVIDIA GPU,
rendering the simulation live in a GLFW window on your desktop.

## Prerequisites

- **NVIDIA GPU** with drivers installed
- **Docker** with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):
  ```bash
  # Install nvidia-container-toolkit (Ubuntu)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
- **Docker Compose** (v2, comes with Docker Desktop or `docker compose` plugin)
- A **HuggingFace token** with access to `lerobot/pi05_base`
- You must be running a **desktop session with X11** (not SSH without forwarding)

## Quick Start

```bash
# 1. Build the image
docker compose build

# 2. Set your HF token
export HF_TOKEN=hf_your_token_here

# 3. Run (opens a live simulation window)
chmod +x run_eval.sh
./run_eval.sh
```

Eval logs and any recorded videos are saved to `./eval_logs/`.

## Configuring the Eval

Edit the `command:` block in `docker-compose.yml` to change tasks, episodes, etc:

| Arg | Default | Notes |
|-----|---------|-------|
| `--env.task` | `libero_10` | Task suite |
| `--env.task_ids` | `[0]` | Which task(s) to run |
| `--eval.n_episodes` | `1` | Episodes per task |
| `--eval.batch_size` | `1` | Parallel envs |
| `--policy.n_action_steps` | `10` | Steps per action chunk |
| `--env.render_mode` | `human` | `human` = live window, `rgb_array` = headless |

## Headless / Recording Mode

To run without a display (e.g. over SSH) and save video instead:

1. In `docker-compose.yml`, change `MUJOCO_GL=glfw` → `MUJOCO_GL=egl`
2. Change `--env.render_mode=human` → `--env.render_mode=rgb_array`
3. You can remove the `DISPLAY` env var and X11 socket volume

## Troubleshooting

**`cannot open display`** — make sure you ran `xhost +local:docker` (the run
script does this for you) and that `$DISPLAY` is set in your shell.

**`Failed to initialize GLFW`** — your GPU driver may not support GLFW in the
container; try switching `MUJOCO_GL=egl` in `docker-compose.yml`.

**`nvidia-smi` not found in container** — the NVIDIA Container Toolkit is not
configured. Follow the install steps above and restart Docker.

**Model download fails** — check your `HF_TOKEN` is valid and you have access
to `lerobot/pi05_base` on HuggingFace Hub.
