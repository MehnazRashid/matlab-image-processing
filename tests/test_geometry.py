import numpy as np

from matlab_image_processing import (
    imresize,
    imrotate,
    imcrop,
    padarray
)


def test_imresize():
    image = np.zeros((100, 100), dtype=np.uint8)

    result = imresize(image, 0.5)

    assert result.shape == (50, 50)


def test_imrotate():
    image = np.zeros((20, 30), dtype=np.uint8)

    result = imrotate(image, 90)

    assert result.shape == (30, 20)


def test_imcrop():
    image = np.zeros((100, 100), dtype=np.uint8)

    result = imcrop(image, [10, 20, 30, 40])

    assert result.shape == (40, 30)


def test_padarray():
    image = np.ones((10, 10), dtype=np.uint8)

    result = padarray(image, 2)

    assert result.shape == (14, 14)