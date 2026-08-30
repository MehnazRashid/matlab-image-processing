import numpy as np

from matlab_image_processing import imhist, histeq, adapthisteq


def test_imhist():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[:50] = 100
    image[50:] = 200

    histogram, bins = imhist(image)

    assert len(histogram) == 256
    assert len(bins) == 257
    assert histogram.sum() == 10000


def test_histeq():
    image = np.full((100, 100), 100, dtype=np.uint8)

    result = histeq(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_adapthisteq():
    image = np.random.randint(
        0,
        256,
        (100, 100),
        dtype=np.uint8,
    )

    result = adapthisteq(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8