import numpy as np

from matlab_image_processing import rgb2gray, rgb2hsv, hsv2rgb


def test_rgb2gray():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    gray = rgb2gray(image)

    assert gray.shape == (10, 10)
    assert gray.dtype == np.uint8
    assert np.all(gray == 76)


def test_rgb2hsv():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    hsv = rgb2hsv(image)

    assert hsv.shape == image.shape
    assert hsv.dtype == np.uint8


def test_hsv2rgb():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :, 0] = 0
    image[:, :, 1] = 255
    image[:, :, 2] = 255

    rgb = hsv2rgb(image)

    assert rgb.shape == image.shape
    assert rgb.dtype == np.uint8