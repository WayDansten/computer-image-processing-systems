from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


Array = np.ndarray
FilterFunction = Callable[[Array, int], Array]


def validate_kernel_size(kernel_size: int) -> None:
    if not isinstance(kernel_size, (int, np.integer)):
        raise TypeError("kernel_size must be an integer")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer greater than or equal to 3")


def median_filter_library(image: Array, kernel_size: int) -> Array:
    validate_kernel_size(kernel_size)
    _validate_image(image)
    if image.ndim == 3 and kernel_size > 5:
        return np.stack(
            [cv2.medianBlur(image[:, :, channel], int(kernel_size)) for channel in range(image.shape[2])],
            axis=2,
        )
    return cv2.medianBlur(image, int(kernel_size))


def median_filter_native(image: Array, kernel_size: int) -> Array:
    validate_kernel_size(kernel_size)
    _validate_image(image)

    radius = kernel_size // 2
    is_grayscale = image.ndim == 2
    height, width = image.shape[:2]
    channels = 1 if is_grayscale else image.shape[2]
    padded = np.pad(
        image,
        ((radius, radius), (radius, radius), (0, 0)) if not is_grayscale
        else ((radius, radius), (radius, radius)),
        mode="edge",
    )
    result = np.empty_like(image)
    window_area = kernel_size * kernel_size
    middle = window_area // 2

    for row in range(height):
        for column in range(width):
            for channel in range(channels):
                if is_grayscale:
                    window = padded[row : row + kernel_size, column : column + kernel_size]
                else:
                    window = padded[
                        row : row + kernel_size,
                        column : column + kernel_size,
                        channel,
                    ]
                values = sorted(window.reshape(-1).tolist())
                value = values[middle]
                if is_grayscale:
                    result[row, column] = value
                else:
                    result[row, column, channel] = value

    return result


def _validate_image(image: Array) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim not in (2, 3):
        raise ValueError("image must be grayscale or multi-channel")
    if image.ndim == 3 and image.shape[2] == 0:
        raise ValueError("image must contain at least one channel")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty")
