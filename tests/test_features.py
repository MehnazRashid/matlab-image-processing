import numpy as np

from matlab_image_processing import (
    extractHOGFeatures,
    detectORBFeatures,
)


def test_extractHOGFeatures():
    image = np.zeros((64, 64), dtype=np.uint8)
    image[20:45, 20:45] = 255

    features = extractHOGFeatures(image)

    assert isinstance(features, np.ndarray)
    assert features.size > 0


def test_detectORBFeatures():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[20:80, 20:80] = 255

    result = detectORBFeatures(image)

    assert isinstance(result, dict)
    assert "Keypoints" in result
    assert "Descriptors" in result
    assert "Responses" in result