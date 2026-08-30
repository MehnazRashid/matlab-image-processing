import numpy as np

from matlab_image_processing import graycomatrix, graycoprops


def test_graycomatrix():
    image = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=np.uint8,
    )

    result = graycomatrix(
        image,
        distances=[1],
        angles=[0],
        levels=2,
    )

    assert result.shape == (2, 2, 1, 1)


def test_graycoprops():
    image = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=np.uint8,
    )

    glcm = graycomatrix(
        image,
        distances=[1],
        angles=[0],
        levels=2,
    )

    contrast = graycoprops(glcm, "contrast")

    assert contrast.shape == (1, 1)
    assert contrast[0, 0] > 0