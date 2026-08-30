import numpy as np

from matlab_image_processing import imadjust, rescale, stretchlim


def test_rescale():
    image = np.array([[0, 50], [100, 200]])

    result = rescale(image)

    assert np.isclose(result.min(), 0)
    assert np.isclose(result.max(), 1)


def test_rescale_custom_range():
    image = np.array([0, 5, 10])

    result = rescale(image, 0, 255)

    assert np.allclose(result, [0, 127.5, 255])


def test_stretchlim():
    image = np.arange(100)

    result = stretchlim(image)

    assert result.shape == (2,)
    assert result[0] < result[1]


def test_imadjust():
    image = np.array([[0, 50], [100, 200]], dtype=np.uint8)

    result = imadjust(image)

    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 255