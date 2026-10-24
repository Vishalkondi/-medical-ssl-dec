"""Decide GPU visibility BEFORE torch is imported.

If a GPU is visible but unusable (old driver, busy device), PyTorch still
touches it in places such as AdamW.step(), which crashes even CPU training.
CUDA_VISIBLE_DEVICES is only honoured before CUDA initialises, so scripts call
`configure_cuda_visibility` first and import torch afterwards. This module
must not import torch.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

log = logging.getLogger(__name__)

_PROBE = "import torch\nif torch.cuda.is_available():\n    torch.zeros(1, device='cuda')\n"


def configure_cuda_visibility(preference: str) -> None:
    if "torch" in sys.modules:
        log.debug("torch already imported; CUDA visibility unchanged")
        return
    if preference == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" is not honoured by every build
        return
    if preference != "auto" or "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=180)
        usable = r.returncode == 0
        reason = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        usable, reason = False, str(exc)
    if not usable:
        log.warning("GPU present but unusable (%s); hiding it and using CPU", reason)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # "" is not honoured by every build
