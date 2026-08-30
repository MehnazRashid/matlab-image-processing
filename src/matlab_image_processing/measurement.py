import numpy as np
from skimage.measure import regionprops as skimage_regionprops


def regionprops(label_image, intensity_image=None):
    """Measure properties of labeled image regions."""
    label_image = np.asarray(label_image)

    regions = skimage_regionprops(
        label_image,
        intensity_image=intensity_image,
    )

    properties = []

    for region in regions:
        properties.append({
            "Area": region.area,
            "Centroid": region.centroid,
            "BoundingBox": region.bbox,
            "Eccentricity": region.eccentricity,
            "Extent": region.extent,
            "MajorAxisLength": region.axis_major_length,
            "MinorAxisLength": region.axis_minor_length,
            "Perimeter": region.perimeter,
            "Solidity": region.solidity,
        })

    return properties