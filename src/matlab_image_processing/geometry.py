import numpy as np
from PIL import Image
from scipy.ndimage import rotate, zoom


import numpy as np
from skimage.transform import resize


def imresize(image, scale):
    """Resize an image using a scale factor or target dimensions."""
    image = np.asarray(image)

    if isinstance(scale, (tuple, list)):
        if len(scale) != 2:
            raise ValueError("Target size must contain height and width.")

        height, width = scale

        if height <= 0 or width <= 0:
            raise ValueError("Target dimensions must be greater than zero.")

        output_shape = (int(height), int(width))

        if image.ndim == 3:
            output_shape += (image.shape[2],)

    else:
        if scale <= 0:
            raise ValueError("Scale must be greater than zero.")

        output_shape = (
            int(image.shape[0] * scale),
            int(image.shape[1] * scale),
        )

        if image.ndim == 3:
            output_shape += (image.shape[2],)

    result = resize(
        image,
        output_shape,
        preserve_range=True,
        anti_aliasing=True,
    )

    if np.issubdtype(image.dtype, np.integer):
        result = np.clip(
            result,
            np.iinfo(image.dtype).min,
            np.iinfo(image.dtype).max,
        ).astype(image.dtype)

    return result


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