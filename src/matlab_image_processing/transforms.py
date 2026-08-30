import numpy as np
from scipy.fft import dct, idct


def fft2(image):
    """Compute the 2D Fourier transform."""
    return np.fft.fft2(image)


def ifft2(image):
    """Compute the inverse 2D Fourier transform."""
    return np.fft.ifft2(image)


def fftshift(image):
    """Shift the zero-frequency component to the center."""
    return np.fft.fftshift(image)


def dct2(image):
    """Compute the 2D discrete cosine transform."""
    image = np.asarray(image, dtype=float)

    return dct(
        dct(image, axis=0, norm="ortho"),
        axis=1,
        norm="ortho",
    )


def idct2(image):
    """Compute the inverse 2D discrete cosine transform."""
    image = np.asarray(image, dtype=float)

    return idct(
        idct(image, axis=0, norm="ortho"),
        axis=1,
        norm="ortho",
    )