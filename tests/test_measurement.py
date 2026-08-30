import numpy as np

from matlab_image_processing import regionprops


def test_regionprops():
    image = np.zeros((20, 20), dtype=int)
    image[5:15, 5:15] = 1

    result = regionprops(image)

    assert len(result) == 1
    assert result[0]["Area"] == 100
    assert len(result[0]["Centroid"]) == 2