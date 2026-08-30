import numpy as np
from scipy.ndimage import sobel, prewitt


def imgradientxy(image, method="sobel"):
    """Calculate horizontal and vertical image gradients."""
    image = np.asarray(image, dtype=float)

    if method.lower() == "sobel":
        gx = sobel(image, axis=1)
        gy = sobel(image, axis=0)
    elif method.lower() == "prewitt":
        gx = prewitt(image, axis=1)
        gy = prewitt(image, axis=0)
    else:
        raise ValueError("Method must be 'sobel' or 'prewitt'.")

    return gx, gy


def imgradient(image, method="sobel"):
    """Calculate gradient magnitude and direction."""
    gx, gy = imgradientxy(image, method)

    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.degrees(np.arctan2(gy, gx))

    return magnitude, direction


def edge(image, method="Canny"):
    """Detect edges in an image."""
    from skimage.feature import canny

    image = np.asarray(image)

    if image.ndim == 3:
        image = np.mean(image, axis=2)

    method = method.lower()

    if method == "canny":
        return canny(image.astype(float))
    elif method == "sobel":
        magnitude, _ = imgradient(image, "sobel")
        threshold = np.mean(magnitude) + np.std(magnitude)
        return magnitude > threshold
    elif method == "prewitt":
        magnitude, _ = imgradient(image, "prewitt")
        threshold = np.mean(magnitude) + np.std(magnitude)
        return magnitude > threshold
    else:
        raise ValueError("Unsupported edge detection method.")