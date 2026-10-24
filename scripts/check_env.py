"""Phase 2 environment check.

Verifies the Python version, required packages, GPU/CUDA, cache locations,
free disk space, the project layout and config, then runs a small DenseNet121
forward pass on the selected device.

Usage (from the project root, with the virtual environment active):
    python scripts/check_env.py
    python scripts/check_env.py --no-pretrained   # skip the weight download
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PACKAGES = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("cv2", "opencv-python-headless"),
    ("PIL", "pillow"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("yaml", "pyyaml"),
    ("tqdm", "tqdm"),
    ("fastapi", "fastapi"),
]
REQUIRED_DIRS = ["configs", "data/raw", "data/processed", "data/splits", "src", "scripts"]
MIN_FREE_GB = 60  # full NIH download (~45 GB) plus resized copies

failures: list[str] = []


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    failures.append(msg)


def check_python() -> None:
    print("\n1. Python")
    v = sys.version_info
    print(f"  {sys.version.split()[0]} at {sys.executable}")
    if (v.major, v.minor) in [(3, 11), (3, 12), (3, 13), (3, 14)]:
        ok("supported version")
    elif v >= (3, 10):
        warn("works in principle; the project is tested against 3.11-3.14")
    else:
        fail("Python 3.11 or newer required")
    if sys.prefix == sys.base_prefix and "google.colab" not in sys.modules:
        warn("not running inside a virtual environment")


def check_packages() -> None:
    print("\n2. Packages")
    for module, pip_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module)
            ok(f"{pip_name} {getattr(mod, '__version__', '?')}")
        except ImportError as exc:
            fail(f"{pip_name} missing ({exc}); run: pip install {pip_name}")


def check_gpu():
    print("\n3. GPU / CUDA")
    try:
        import torch
    except ImportError:
        fail("torch not importable, skipping GPU check")
        return None
    print(f"  torch built with CUDA: {torch.version.cuda}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        ok(f"{props.name}, {props.total_memory / 1024**3:.1f} GB VRAM, "
           f"compute capability {props.major}.{props.minor}")
        return torch.device("cuda")
    if torch.version.cuda is None:
        warn("CPU-only torch build installed; reinstall with a CUDA wheel to use the GPU")
    else:
        warn("CUDA build installed but no GPU visible; check the NVIDIA driver (nvidia-smi)")
    return torch.device("cpu")


def check_storage() -> None:
    print("\n4. Storage and caches")
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
    msg = f"{free_gb:.0f} GB free on the project drive ({PROJECT_ROOT.anchor})"
    ok(msg) if free_gb >= MIN_FREE_GB else warn(msg + f"; >= {MIN_FREE_GB} GB recommended")
    for var in ["PIP_CACHE_DIR", "TORCH_HOME"]:
        value = os.environ.get(var)
        if value:
            ok(f"{var} = {value}")
        elif platform.system() == "Windows":
            warn(f"{var} not set; downloads will go to the C: drive")


def check_project() -> None:
    print("\n5. Project layout and config")
    for d in REQUIRED_DIRS:
        (ok if (PROJECT_ROOT / d).is_dir() else fail)(f"{d}/")
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    try:
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ok(f"config loaded: task={cfg['task']['classes']}, seed={cfg['project']['seed']}")
    except Exception as exc:  # noqa: BLE001 - report any config problem
        fail(f"could not load {cfg_path}: {exc}")


def check_forward(device, pretrained: bool) -> None:
    print("\n6. DenseNet121 forward pass")
    if device is None:
        return
    import torch
    from torchvision.models import DenseNet121_Weights, densenet121

    try:
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights).to(device).eval()
        ok("ImageNet weights loaded" if pretrained else "random init (download skipped)")
    except Exception as exc:  # noqa: BLE001 - network/cache errors vary
        fail(f"model creation failed: {exc}")
        return

    batch = torch.randn(8, 3, 224, 224, device=device)
    use_amp = device.type == "cuda"
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
            start = time.perf_counter()
            out = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        assert out.shape == (8, 1000), f"unexpected output shape {tuple(out.shape)}"
        ok(f"output {tuple(out.shape)} on {device} in {elapsed:.2f}s (AMP={use_amp})")
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**3
            ok(f"peak GPU memory {peak:.2f} GB")
    except Exception as exc:  # noqa: BLE001
        fail(f"forward pass failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="skip downloading ImageNet weights")
    args = parser.parse_args()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Platform: {platform.platform()}")
    check_python()
    check_packages()
    device = check_gpu()
    check_storage()
    check_project()
    check_forward(device, pretrained=not args.no_pretrained)

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S):"))
    for f in failures:
        print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
