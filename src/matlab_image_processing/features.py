import numpy as np
from skimage.feature import hog, ORB, SIFT


def extractHOGFeatures(image, **kwargs):
    """Extract Histogram of Oriented Gradients features."""
    image = np.asarray(image)

    if image.ndim == 3:
        image = np.mean(image, axis=2)

    features = hog(
        image,
        **kwargs,
    )

    return features


def detectORBFeatures(image, **kwargs):
    """Detect and extract ORB features."""
    image = np.asarray(image)

    if image.ndim == 3:
        image = np.mean(image, axis=2)

    detector = ORB(**kwargs)
    detector.detect_and_extract(image)

    return {
        "Keypoints": detector.keypoints,
        "Descriptors": detector.descriptors,
        "Responses": detector.responses,
    }


def detectSURFFeatures(image, **kwargs):
    """Detect SURF features."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for SURF features."
        ) from exc

    image = np.asarray(image)

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = image.astype(np.uint8)

    detector = cv2.xfeatures2d.SURF_create(**kwargs)
    keypoints, descriptors = detector.detectAndCompute(image, None)

    return {
        "Keypoints": keypoints,
        "Descriptors": descriptors,
    }