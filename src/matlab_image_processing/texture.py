import numpy as np
from skimage.feature import graycomatrix, graycoprops


def graycomatrix(
    image,
    distances=[1],
    angles=[0],
    levels=256,
    symmetric=False,
    normed=False,
):
    """Create a gray-level co-occurrence matrix."""
    image = np.asarray(image)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, levels - 1).astype(np.uint8)

    return _graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=symmetric,
        normed=normed,
    )


def graycoprops(glcm, property_name):
    """Calculate texture properties from a GLCM."""
    return _graycoprops(glcm, property_name)


from skimage.feature import (
    graycomatrix as _graycomatrix,
    graycoprops as _graycoprops,
)