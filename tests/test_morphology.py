import numpy as np

from matlab_image_processing import imerode, imdilate, imopen, imclose


def test_imerode():
    image = np.ones((10, 10), dtype=bool)

    result = imerode(image)

    assert result.shape == image.shape
    assert result.sum() < image.sum()


def test_imdilate():
    image = np.zeros((10, 10), dtype=bool)
    image[5, 5] = True

    result = imdilate(image)

    assert result.sum() > 1


def test_imopen():
    image = np.zeros((10, 10), dtype=bool)
    image[3:7, 3:7] = True

    result = imopen(image)

    assert result.shape == image.shape


def test_imclose():
    image = np.ones((10, 10), dtype=bool)
    image[4:6, 4:6] = False

    result = imclose(image)

    assert result.shape == image.shape