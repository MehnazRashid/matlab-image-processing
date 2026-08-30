import numpy as np
from PIL import Image
from scipy.ndimage import rotate, zoom


def imresize(image, scale):
    """Resize an image by a scale factor."""
    image = np.asarray(image)

    if scale <= 0:
        raise ValueError("Scale must be greater than zero.")

    if scale == 1:
        return image.copy()

    if image.ndim == 2:
        factors = (scale, scale)
    elif image.ndim == 3:
        factors = (scale, scale, 1)
    else:
        raise ValueError("Image must be 2D or 3D.")

    return zoom(image, factors, order=1)


def imrotate(image, angle):
    """Rotate an image by an angle in degrees."""
    image = np.asarray(image)

    return rotate(
        image,
        angle,
        reshape=True,
        order=1,
        mode="constant",
        cval=0
    )


def imcrop(image, rect):
    """Crop an image using [x, y, width, height]."""
    image = np.asarray(image)

    x, y, width, height = rect

    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)

    return image[y:y + height, x:x + width]


def padarray(image, pad_width, mode="constant", constant_values=0):
    """Pad an image."""
    image = np.asarray(image)

    return np.pad(
        image,
        pad_width,
        mode=mode,
        constant_values=constant_values
        if mode == "constant"
        else None
    )