import numpy as np
from scipy.ndimage import label, binary_erosion
from skimage.measure import label as sklabel


def imbinarize(image, threshold=None):
    """Convert an image to a binary image."""
    image = np.asarray(image)

    if threshold is None:
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(image)

    return image > threshold


def bwareaopen(image, area):
    """Remove connected components smaller than a given area."""
    image = np.asarray(image, dtype=bool)

    labels, count = label(image)

    sizes = np.bincount(labels.ravel())

    result = np.zeros_like(image)

    for i in range(1, count + 1):
        if sizes[i] >= area:
            result[labels == i] = True

    return result


def bwperim(image):
    """Find the perimeter of objects in a binary image."""
    image = np.asarray(image, dtype=bool)

    eroded = binary_erosion(image)

    return image & ~eroded


def bwlabel(image, connectivity=8):
    """Label connected components in a binary image."""
    image = np.asarray(image, dtype=bool)

    if connectivity == 4:
        connectivity_structure = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=int,
        )
    elif connectivity == 8:
        connectivity_structure = np.ones((3, 3), dtype=int)
    else:
        raise ValueError("Connectivity must be 4 or 8.")

    return label(image, structure=connectivity_structure)


def bwconncomp(image, connectivity=8):
    """Find connected components in a binary image."""
    labels, count = bwlabel(image, connectivity)

    components = []

    for i in range(1, count + 1):
        components.append(np.argwhere(labels == i))

    return {
        "NumObjects": count,
        "PixelIdxList": components,
    }