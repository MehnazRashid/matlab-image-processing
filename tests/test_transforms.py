import numpy as np

from matlab_image_processing import fft2, ifft2, fftshift, dct2, idct2


def test_fft2_ifft2():
    image = np.random.rand(20, 20)

    transformed = fft2(image)
    reconstructed = ifft2(transformed)

    assert transformed.shape == image.shape
    assert np.allclose(reconstructed.real, image)


def test_fftshift():
    image = np.zeros((10, 10))
    image[0, 0] = 1

    result = fftshift(image)

    assert result[5, 5] == 1


def test_dct2_idct2():
    image = np.random.rand(20, 20)

    transformed = dct2(image)
    reconstructed = idct2(transformed)

    assert transformed.shape == image.shape
    assert np.allclose(reconstructed, image)