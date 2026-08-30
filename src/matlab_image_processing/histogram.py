import numpy as np
from skimage import exposure


def imhist(image, nbins=256):
    """Compute the histogram of an image."""
    image = np.asarray(image)

    if image.ndim == 3:
        image = np.mean(image, axis=2)

    histogram, bin_edges = np.histogram(
        image.ravel(),
        bins=nbins,
        range=(0, 255) if image.dtype == np.uint8 else None,
    )

    return histogram, bin_edges


def histeq(image, nbins=256):
    """Enhance image contrast using histogram equalization."""
    image = np.asarray(image)

    if image.dtype == np.uint8:
        result = exposure.equalize_hist(image, nbins=nbins)
        return np.round(result * 255).astype(np.uint8)

    return exposure.equalize_hist(image, nbins=nbins)


def adapthisteq(image, clip_limit=0.01):
    """Enhance local image contrast using adaptive histogram equalization."""
    image = np.asarray(image)

    if image.dtype == np.uint8:
        result = exposure.equalize_adapthist(
            image,
            clip_limit=clip_limit,
        )
        return np.round(result * 255).astype(np.uint8)

    return exposure.equalize_adapthist(
        image,
        clip_limit=clip_limit,
    )