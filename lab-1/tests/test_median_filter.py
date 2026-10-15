import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from median_filter import median_filter_library, median_filter_native, validate_kernel_size


@pytest.mark.parametrize("kernel_size", [0, 1, 2, 4, -3])
def test_kernel_size_must_be_odd_and_at_least_three(kernel_size):
    with pytest.raises(ValueError):
        validate_kernel_size(kernel_size)


def test_kernel_size_must_be_integer():
    with pytest.raises(TypeError):
        validate_kernel_size(3.5)


@pytest.mark.parametrize("shape", [(8, 9), (8, 9, 3)])
@pytest.mark.parametrize("kernel_size", [3, 5])
def test_native_matches_opencv(shape, kernel_size):
    image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    expected = cv2.medianBlur(image, kernel_size)
    actual = median_filter_native(image, kernel_size)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("filter_function", [median_filter_library, median_filter_native])
def test_filter_preserves_shape_and_dtype(filter_function):
    image = np.zeros((7, 6, 3), dtype=np.uint8)
    result = filter_function(image, 3)
    assert result.shape == image.shape
    assert result.dtype == image.dtype


@pytest.mark.parametrize("image", [np.zeros((4, 4), dtype=np.float32), np.zeros((4,), dtype=np.uint8)])
def test_filter_rejects_unsupported_images(image):
    with pytest.raises(ValueError):
        median_filter_native(image, 3)
