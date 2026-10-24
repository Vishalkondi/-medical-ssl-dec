"""Image loading, validation and resizing.

Used offline (scripts/prepare_images.py) and, in Phase 7, by the backend, so
training and inference share one definition of preprocessing.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageValidationError(ValueError):
    """Raised when an image file is unreadable or has unusable content."""


def load_grayscale(path: str | Path) -> np.ndarray:
    """Read an image file as a 2-D uint8 array (H, W).

    Handles 8/16-bit, RGB and RGBA inputs (a few NIH files are RGBA).
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageValidationError(f"unreadable or corrupt image: {path}")
    return to_grayscale_uint8(img)


def decode_grayscale(data: bytes) -> np.ndarray:
    """Decode in-memory image bytes (e.g. an upload) to a 2-D uint8 array."""
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageValidationError("unreadable or corrupt image data")
    return to_grayscale_uint8(img)


def to_grayscale_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 1:
            img = img[:, :, 0]
        else:
            raise ImageValidationError(f"unsupported channel count {img.shape[2]}")
    if img.ndim != 2:
        raise ImageValidationError(f"unexpected image shape {img.shape}")
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif img.dtype != np.uint8:
        raise ImageValidationError(f"unsupported dtype {img.dtype}")
    return img


def validate_content(img: np.ndarray, min_side: int = 64) -> None:
    """Reject images that are too small or blank (constant intensity)."""
    h, w = img.shape
    if min(h, w) < min_side:
        raise ImageValidationError(f"image too small ({w}x{h}), minimum side {min_side}px")
    if int(img.max()) == int(img.min()):
        raise ImageValidationError("image has constant intensity (blank)")


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    """Resize to size x size. Non-square inputs are centre-padded first so
    anatomy is not stretched. INTER_AREA is the right filter for downscaling."""
    h, w = img.shape
    if h != w:
        side = max(h, w)
        top, left = (side - h) // 2, (side - w) // 2
        img = cv2.copyMakeBorder(img, top, side - h - top, left, side - w - left,
                                 cv2.BORDER_CONSTANT, value=0)
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_LINEAR
    out = cv2.resize(img, (size, size), interpolation=interp)
    assert out.shape == (size, size), out.shape
    return out


def to_model_input(img: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    """uint8 (H, W) -> normalised float32 (3, H, W). The grayscale channel is
    repeated so ImageNet-pretrained backbones can be used unchanged."""
    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(f"expected uint8 (H, W), got {img.dtype} {img.shape}")
    x = img.astype(np.float32) / 255.0
    x = np.repeat(x[None], 3, axis=0)
    m = np.asarray(mean, np.float32)[:, None, None]
    s = np.asarray(std, np.float32)[:, None, None]
    return (x - m) / s


def inference_preprocess(gray: np.ndarray, preprocess_size: int, image_size: int,
                         mean: list[float], std: list[float]) -> np.ndarray:
    """Exactly the evaluation pipeline used in training: the offline resize to
    `preprocess_size` (prepare_images.py), then resize to `image_size`."""
    return to_model_input(resize_square(resize_square(gray, preprocess_size), image_size), mean, std)
