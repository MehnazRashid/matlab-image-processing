import numpy as np

from matlab_image_processing import (
    imbinarize,
    bwareaopen,
    bwperim,
    bwlabel,
    bwconncomp,
)


def test_imbinarize():
    image = np.array([[0, 50], [200, 255]])

    result = imbinarize(image)

    assert result.dtype == bool
    assert result.shape == image.shape


def test_bwareaopen():
    image = np.zeros((20, 20), dtype=bool)
    image[1, 1] = True
    image[10:15, 10:15] = True

    result = bwareaopen(image, 5)

    assert not result[1, 1]
    assert result[12, 12]


def test_bwperim():
    image = np.zeros((10, 10), dtype=bool)
    image[2:8, 2:8] = True

    result = bwperim(image)

    assert result.shape == image.shape
    assert result.sum() > 0


def test_bwlabel():
    image = np.zeros((10, 10), dtype=bool)
    image[1:3, 1:3] = True
    image[7:9, 7:9] = True

    labels, count = bwlabel(image)

    assert count == 2
    assert labels.max() == 2


def test_bwconncomp():
    image = np.zeros((10, 10), dtype=bool)
    image[1:3, 1:3] = True
    image[7:9, 7:9] = True

    result = bwconncomp(image)

    assert result["NumObjects"] == 2
    assert len(result["PixelIdxList"]) == 2