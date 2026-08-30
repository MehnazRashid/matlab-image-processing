import numpy as np


def rescale(image, new_min=0, new_max=1):
    """Rescale image values to a new range."""
    image = np.asarray(image, dtype=float)

    old_min = np.min(image)
    old_max = np.max(image)

    if old_max == old_min:
        return np.full_like(image, new_min)

    return (
        (image - old_min)
        / (old_max - old_min)
        * (new_max - new_min)
        + new_min
    )


def stretchlim(image, tol=0.01):
    """Find lower and upper intensity limits."""
    image = np.asarray(image)

    if image.ndim == 3:
        image = image.reshape(-1, image.shape[-1])

    lower = np.percentile(image, tol * 100)
    upper = np.percentile(image, (1 - tol) * 100)

    return np.array([lower, upper])


def imadjust(image, in_range=None, out_range=None, gamma=1.0):
    """Adjust image intensity values."""
    image = np.asarray(image)

    if in_range is None:
        in_range = [np.min(image), np.max(image)]

    if out_range is None:
        if np.issubdtype(image.dtype, np.integer):
            out_range = [0, np.iinfo(image.dtype).max]
        else:
            out_range = [0.0, 1.0]

    in_min, in_max = in_range
    out_min, out_max = out_range

    image_float = image.astype(float)

    if in_max == in_min:
        adjusted = np.full_like(image_float, out_min)
    else:
        adjusted = (
            (image_float - in_min)
            / (in_max - in_min)
        )

        adjusted = np.clip(adjusted, 0, 1)
        adjusted = adjusted ** gamma
        adjusted = adjusted * (out_max - out_min) + out_min

    if np.issubdtype(image.dtype, np.integer):
        adjusted = np.clip(
            adjusted,
            np.iinfo(image.dtype).min,
            np.iinfo(image.dtype).max,
        ).astype(image.dtype)

    return adjusted