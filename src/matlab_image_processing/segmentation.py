import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from skimage.segmentation import watershed as skimage_watershed
from scipy import ndimage


def graythresh(image):
    """Calculate an Otsu threshold."""
    image = np.asarray(image)

    return threshold_otsu(image)


def adaptthresh(image, neighborhood_size=51, sensitivity=0.5):
    """Calculate a locally adaptive threshold."""
    image = np.asarray(image, dtype=float)

    threshold = threshold_local(
        image,
        block_size=neighborhood_size,
        offset=(0.5 - sensitivity),
    )

    return threshold


def watershed(image, markers=None):
    """Perform watershed segmentation."""
    image = np.asarray(image)

    if markers is None:
        markers, _ = ndimage.label(image)

    return skimage_watershed(image, markers=markers)