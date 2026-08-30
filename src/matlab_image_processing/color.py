import numpy as np
from PIL import Image


def rgb2gray(image):
    """Convert an RGB image to grayscale."""
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be an RGB image.")

    gray = (
        0.2989 * image[:, :, 0]
        + 0.5870 * image[:, :, 1]
        + 0.1140 * image[:, :, 2]
    )

    return gray.astype(image.dtype)


def rgb2hsv(image):
    """Convert an RGB image to HSV."""
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be an RGB image.")

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    hsv = np.asarray(Image.fromarray(image).convert("HSV"))

    return hsv


def hsv2rgb(image):
    """Convert an HSV image to RGB."""
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be an HSV image.")

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    rgb = np.asarray(Image.fromarray(image, mode="HSV").convert("RGB"))

    return rgb