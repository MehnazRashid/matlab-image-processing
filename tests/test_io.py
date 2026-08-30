from matlab_image_processing import imread, imwrite, imshow, imfinfo
from PIL import Image
import numpy as np


def test_imread():
    original = np.zeros((10, 10, 3), dtype=np.uint8)
    original[2:8, 2:8] = [255, 0, 0]

    Image.fromarray(original).save("test_image.png")

    result = imread("test_image.png")

    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10, 3)
    assert np.array_equal(result, original)


def test_imwrite():
    original = np.zeros((10, 10, 3), dtype=np.uint8)
    original[2:8, 2:8] = [0, 255, 0]

    imwrite("output_test.png", original)

    result = np.array(Image.open("output_test.png"))

    assert np.array_equal(result, original)


def test_imfinfo():
    original = np.zeros((20, 30, 3), dtype=np.uint8)

    Image.fromarray(original).save("info_test.png")

    info = imfinfo("info_test.png")

    assert info["format"] == "PNG"
    assert info["width"] == 30
    assert info["height"] == 20
    assert info["mode"] == "RGB"


def test_imshow():
    image = np.zeros((10, 10), dtype=np.uint8)

    result = imshow(image)

    assert result is None