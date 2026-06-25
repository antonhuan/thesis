"""
Remote VLM planner client — drop-in replacement for VLMPlanner.

Calls the VLM inference server over HTTP instead of running the model locally.

Usage in vlm_robot_orchestrator.py:
    from vlm_client import RemoteVLMPlanner

    # Replace:
    #   planner = VLMPlanner(cfg.vlm_model, temperature=cfg.vlm_temperature)
    # With:
    #   planner = RemoteVLMPlanner("http://10.35.9.XX:9090", temperature=cfg.vlm_temperature)
"""

import base64
import io
import logging

import requests
from PIL import Image


class RemoteVLMPlanner:
    """Calls a remote VLM server for decomposition and evaluation."""

    def __init__(self, server_url: str, temperature: float = 0.7, timeout: float = 60.0):
        self.server_url = server_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

        # Verify connection
        try:
            r = requests.get(f"{self.server_url}/health", timeout=5)
            r.raise_for_status()
            info = r.json()
            logging.info(f"Connected to VLM server: {info}")
        except Exception as e:
            raise ConnectionError(
                f"Could not reach VLM server at {self.server_url}: {e}"
            )

    def _encode_frame(self, frame: Image.Image | None) -> str | None:
        if frame is None:
            return None
        buf = io.BytesIO()
        frame.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def decompose(self, prompt: str, frame: Image.Image | None) -> list[str]:
        """High-level prompt + observation -> ordered list of sub-tasks."""
        payload = {
            "prompt": prompt,
            "image_b64": self._encode_frame(frame),
            "temperature": self.temperature,
        }

        r = requests.post(
            f"{self.server_url}/decompose",
            json=payload,
            timeout=self.timeout,
        )

        if r.status_code == 422:
            detail = r.json().get("detail", "Unknown parsing error")
            raise ValueError(detail)

        r.raise_for_status()
        data = r.json()

        logging.info(f"Raw VLM decomposition output: {data.get('raw_output', '')!r}")
        return data["subtasks"]

    def evaluate(self, sub_task: str, frame: Image.Image | None) -> dict:
        """Observation + attempted sub-task -> {'success': bool, 'reason': str}."""
        payload = {
            "sub_task": sub_task,
            "image_b64": self._encode_frame(frame),
            "temperature": self.temperature,
        }

        r = requests.post(
            f"{self.server_url}/evaluate",
            json=payload,
            timeout=self.timeout,
        )

        if r.status_code == 422:
            detail = r.json().get("detail", "Unknown parsing error")
            raise ValueError(detail)

        r.raise_for_status()
        data = r.json()

        logging.info(f"Raw VLM evaluation output: {data.get('raw_output', '')!r}")
        return {"success": data["success"], "reason": data["reason"]}