import numpy as np


def imadd(image1, image2):
    """Add two images."""
    return np.add(image1, image2)


def imsubtract(image1, image2):
    """Subtract one image from another."""
    return np.subtract(image1, image2)


def immultiply(image1, image2):
    """Multiply two images."""
    return np.multiply(image1, image2)


def imdivide(image1, image2):
    """Divide one image by another."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(image1, image2)


def imcomplement(image):
    """Return the complement of an image."""
    image = np.asarray(image)

    if np.issubdtype(image.dtype, np.integer):
        return np.iinfo(image.dtype).max - image

    return 1.0 - image


def imabsdiff(image1, image2):
    """Return the absolute difference between two images."""
    return np.abs(np.subtract(image1, image2))