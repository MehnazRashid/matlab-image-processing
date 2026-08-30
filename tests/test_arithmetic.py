import numpy as np

from matlab_image_processing import (
    imadd,
    imsubtract,
    immultiply,
    imdivide,
    imcomplement,
    imabsdiff,
)


def test_imadd():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])

    assert np.array_equal(imadd(a, b), [[6, 8], [10, 12]])


def test_imsubtract():
    a = np.array([[5, 6], [7, 8]])
    b = np.array([[1, 2], [3, 4]])

    assert np.array_equal(imsubtract(a, b), [[4, 4], [4, 4]])


def test_immultiply():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[2, 3], [4, 5]])

    assert np.array_equal(immultiply(a, b), [[2, 6], [12, 20]])


def test_imdivide():
    a = np.array([[2, 4], [6, 8]])
    b = np.array([[2, 2], [2, 2]])

    assert np.array_equal(imdivide(a, b), [[1, 2], [3, 4]])


def test_imcomplement():
    image = np.array([[0, 100], [200, 255]], dtype=np.uint8)

    result = imcomplement(image)

    assert np.array_equal(result, [[255, 155], [55, 0]])


def test_imabsdiff():
    a = np.array([[1, 5], [10, 20]])
    b = np.array([[4, 2], [6, 30]])

    assert np.array_equal(imabsdiff(a, b), [[3, 3], [4, 10]])