import numpy as np
from skimage.util import random_noise


def imnoise(image, noise_type="gaussian", **kwargs):
    """Add noise to an image."""
    image = np.asarray(image)

    original_dtype = image.dtype

    if np.issubdtype(original_dtype, np.integer):
        image_float = image.astype(float) / np.iinfo(original_dtype).max
    else:
        image_float = image.astype(float)

    noisy = random_noise(
        image_float,
        mode=noise_type,
        **kwargs,
    )

    if np.issubdtype(original_dtype, np.integer):
        noisy = np.clip(
            noisy * np.iinfo(original_dtype).max,
            0,
            np.iinfo(original_dtype).max,
        ).astype(original_dtype)

    return noisy