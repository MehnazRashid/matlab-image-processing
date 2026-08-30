import numpy as np
from scipy.ndimage import convolve, gaussian_filter, median_filter


def imfilter(image, kernel):
    """Apply a 2D filter kernel to an image."""
    image = np.asarray(image)
    kernel = np.asarray(kernel)

    if image.ndim == 2:
        return convolve(image, kernel, mode="reflect")

    if image.ndim == 3:
        result = np.empty_like(image)

        for channel in range(image.shape[2]):
            result[:, :, channel] = convolve(
                image[:, :, channel],
                kernel,
                mode="reflect",
            )

        return result

    raise ValueError("Image must be 2D or 3D.")


def imgaussfilt(image, sigma=0.5):
    """Apply a Gaussian filter to an image."""
    image = np.asarray(image)

    if sigma <= 0:
        raise ValueError("Sigma must be greater than zero.")

    return gaussian_filter(image, sigma=sigma)


def medfilt2(image, kernel_size=3):
    """Apply a 2D median filter to an image."""
    image = np.asarray(image)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if len(kernel_size) != 2:
        raise ValueError("Kernel size must have two dimensions.")

    return median_filter(image, size=kernel_size)