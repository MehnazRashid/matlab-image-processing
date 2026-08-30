import numpy as np

from matlab_image_processing import imfilter, imgaussfilt, medfilt2


def test_imfilter():
    image = np.ones((10, 10))
    kernel = np.ones((3, 3))

    result = imfilter(image, kernel)

    assert result.shape == image.shape


def test_imgaussfilt():
    image = np.zeros((20, 20))
    image[10, 10] = 255

    result = imgaussfilt(image, 2)

    assert result.shape == image.shape
    assert result[10, 10] < 255
    assert result[10, 10] > 0


def test_medfilt2():
    image = np.ones((10, 10))
    image[5, 5] = 100

    result = medfilt2(image, 3)

    assert result.shape == image.shape
    assert result[5, 5] == 1