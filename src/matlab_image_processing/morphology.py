import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation


def _structuring_element(se):
    if se is None:
        return np.ones((3, 3), dtype=bool)

    if isinstance(se, int):
        return np.ones((se, se), dtype=bool)

    return np.asarray(se, dtype=bool)


def imerode(image, se=None):
    """Erode a binary image."""
    image = np.asarray(image, dtype=bool)
    se = _structuring_element(se)

    return binary_erosion(image, structure=se)


def imdilate(image, se=None):
    """Dilate a binary image."""
    image = np.asarray(image, dtype=bool)
    se = _structuring_element(se)

    return binary_dilation(image, structure=se)


def imopen(image, se=None):
    """Perform morphological opening."""
    return imdilate(imerode(image, se), se)


def imclose(image, se=None):
    """Perform morphological closing."""
    return imerode(imdilate(image, se), se)