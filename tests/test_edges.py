import numpy as np

from matlab_image_processing import edge, imgradient, imgradientxy


def test_imgradientxy():
    image = np.zeros((50, 50))
    image[:, 25:] = 255

    gx, gy = imgradientxy(image)

    assert gx.shape == image.shape
    assert gy.shape == image.shape


def test_imgradient():
    image = np.zeros((50, 50))
    image[:, 25:] = 255

    magnitude, direction = imgradient(image)

    assert magnitude.shape == image.shape
    assert direction.shape == image.shape


def test_edge():
    image = np.zeros((50, 50))
    image[20:30, 20:30] = 255

    result = edge(image, "Canny")

    assert result.shape == image.shape
    assert result.dtype == bool