import cv2
import numpy as np
import pytest

from src.data.preprocessing import (
    ImageValidationError, decode_grayscale, load_grayscale, resize_square, validate_content,
)


def gradient(h, w, dtype=np.uint8):
    img = np.tile(np.linspace(0, 255, w), (h, 1))
    return img.astype(dtype)


def test_resize_square_output_shape():
    assert resize_square(gradient(1024, 1024), 256).shape == (256, 256)
    assert resize_square(gradient(300, 500), 256).shape == (256, 256)


def test_rgba_and_16bit_become_uint8_gray(tmp_path):
    rgba = cv2.cvtColor(gradient(100, 100), cv2.COLOR_GRAY2BGRA)
    p = tmp_path / "rgba.png"
    cv2.imwrite(str(p), rgba)
    out = load_grayscale(p)
    assert out.ndim == 2 and out.dtype == np.uint8

    p16 = tmp_path / "16.png"
    cv2.imwrite(str(p16), (gradient(100, 100).astype(np.uint16) * 257))
    out16 = load_grayscale(p16)
    assert out16.dtype == np.uint8 and out16.max() == 255


def test_corrupt_bytes_rejected():
    with pytest.raises(ImageValidationError):
        decode_grayscale(b"not an image at all")


def test_corrupt_file_rejected(tmp_path):
    p = tmp_path / "broken.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n garbage")
    with pytest.raises(ImageValidationError):
        load_grayscale(p)


def test_blank_and_tiny_images_rejected():
    with pytest.raises(ImageValidationError, match="constant"):
        validate_content(np.full((128, 128), 7, np.uint8))
    with pytest.raises(ImageValidationError, match="too small"):
        validate_content(gradient(32, 32))
    validate_content(gradient(128, 128))  # valid image passes
