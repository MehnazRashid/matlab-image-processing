import numpy as np

from matlab_image_processing import imnoise


def test_imnoise_gaussian():
    image = np.full((100, 100), 128, dtype=np.uint8)

    result = imnoise(
        image,
        "gaussian",
        var=0.01,
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert not np.array_equal(result, image)


def test_imnoise_salt_pepper():
    image = np.full((100, 100), 128, dtype=np.uint8)

    result = imnoise(
        image,
        "s&p",
        amount=0.1,
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8