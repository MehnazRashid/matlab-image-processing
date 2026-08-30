import numpy as np

from matlab_image_processing import graythresh, adaptthresh, watershed


def test_graythresh():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[:50] = 50
    image[50:] = 200

    threshold = graythresh(image)

    assert 50 <= threshold <= 200


def test_adaptthresh():
    image = np.random.randint(
        0,
        256,
        (50, 50),
        dtype=np.uint8,
    )

    result = adaptthresh(image)

    assert result.shape == image.shape


def test_watershed():
    image = np.zeros((50, 50), dtype=float)
    image[10:40, 10:40] = 1

    markers = np.zeros((50, 50), dtype=int)
    markers[20, 20] = 1
    markers[30, 30] = 2

    result = watershed(image, markers)

    assert result.shape == image.shape
    assert result.max() >= 2