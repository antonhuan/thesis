# fast_policy_server.py
#
# Drop-in replacement for `python -m lerobot.async_inference.policy_server` that
# removes the ~3 minute stall when loading a pi05 checkpoint.
#
# Why it is slow upstream: PI05Policy.from_pretrained builds the model with
# `cls(config)` BEFORE reading the checkpoint, and PaliGemmaWithExpertModel
# constructs PaliGemma + the action expert from a config object rather than via
# HF's from_pretrained. That runs a full random init over 4.14B parameters in
# float32 on CPU — measured at 178s — and every one of those values is then
# overwritten by the checkpoint anyway. HF's own from_pretrained avoids this by
# building under a meta device; here we get the same effect with the
# `no_init_weights` context manager, which swaps the torch.nn.init primitives
# for no-ops. Construction drops from 178s to 3.3s; buffers (rotary inv_freq
# and friends) are still built normally, so nothing needs re-materialising.
#
# Usage (inside the lerobot-so101 container, same flags as upstream):
#   python fast_policy_server.py --host=0.0.0.0 --port=8080

import logging
import threading

import torch

from transformers.initialization import no_init_weights

from lerobot.async_inference.policy_server import serve, PolicyServer
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def _assert_weights_loaded(policy) -> None:
    """Fail loudly if the checkpoint did not actually land in the model.

    PI05Policy.from_pretrained wraps its whole load in a bare `except Exception`
    that prints a warning and returns the model regardless. Upstream that leaves
    you serving random weights; with the init skipped it leaves you serving
    uninitialised memory. Either way it is a silent wrong-model bug, so check.
    """
    empty = []
    for name, param in policy.named_parameters():
        if param.numel() == 0:
            continue
        as_float = param.detach().float()
        if not torch.isfinite(as_float).all() or as_float.abs().max().item() == 0.0:
            empty.append(name)
            if len(empty) >= 5:
                break

    if empty:
        raise RuntimeError(
            "pi05 checkpoint was not loaded: "
            f"{len(empty)}+ parameters are all-zero or non-finite (e.g. {empty[:5]}). "
            "Check the preceding from_pretrained output for the swallowed error."
        )


_original_from_pretrained = PI05Policy.from_pretrained.__func__


def _fast_from_pretrained(cls, *args, **kwargs):
    """PI05Policy.from_pretrained without the throwaway random init."""
    with no_init_weights():
        policy = _original_from_pretrained(cls, *args, **kwargs)
    _assert_weights_loaded(policy)
    return policy


PI05Policy.from_pretrained = classmethod(_fast_from_pretrained)


# ---------------------------------------------------------------------------
# Fix: reset per-episode server state between back-to-back episodes.
#
# The upstream PolicyServer accumulates _predicted_timesteps and
# last_processed_obs across episodes (they are only cleared on Ready(),
# i.e. a new client connection). When the orchestrator runs episodes
# back-to-back, the client restarts timesteps at 0 but the server still
# holds the previous episode's set — causing it to silently skip new
# observations whose timesteps collide with old ones. The VLA then
# produces actions from a single first-frame inference and the arm
# barely moves.
#
# The fix: when a must_go observation with timestep 0 arrives (the
# client signals a new episode), clear the stale state.
# ---------------------------------------------------------------------------
_logger = logging.getLogger("fast_policy_server")

_original_enqueue = PolicyServer._enqueue_observation


def _patched_enqueue_observation(self, obs):
    if obs.must_go and obs.get_timestep() == 0:
        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()
        self.last_processed_obs = None
        _logger.info("Episode state reset (new must_go ts=0 received)")
    return _original_enqueue(self, obs)


PolicyServer._enqueue_observation = _patched_enqueue_observation


if __name__ == "__main__":
    # draccus-wrapped: --host, --port, --fps, --obs_queue_timeout, ... all pass through.
    serve()
