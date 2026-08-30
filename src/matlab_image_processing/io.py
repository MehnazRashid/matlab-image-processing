from PIL import Image
import numpy as np


def imread(filename):
    """Read an image and return it as a NumPy array."""
    image = Image.open(filename)
    return np.array(image)


def imwrite(filename, image):
    """Write a NumPy array to an image file."""
    Image.fromarray(image).save(filename)


def imshow(image, title=None):
    """Display an image."""
    import matplotlib.pyplot as plt

    plt.imshow(image, cmap="gray" if image.ndim == 2 else None)

    if title is not None:
        plt.title(title)

    plt.axis("off")
    plt.show()


def imfinfo(filename):
    """Return basic information about an image."""
    image = Image.open(filename)

    return {
        "filename": filename,
        "format": image.format,
        "mode": image.mode,
        "size": image.size,
        "width": image.width,
        "height": image.height,
    }