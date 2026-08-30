# MATLAB Image Processing

A Python image processing library inspired by MATLAB's Image Processing Toolbox.

## Table of Contents

- [I/O](#io)
- [Color](#color)
- [Geometry](#geometry)
- [Arithmetic](#arithmetic)
- [Histogram](#histogram)
- [Filtering](#filtering)
- [Edges](#edges)
- [Morphology](#morphology)
- [Binary Images](#binary-images)
- [Measurement](#measurement)
- [Segmentation](#segmentation)
- [Texture](#texture)
- [Transforms](#transforms)
- [Noise](#noise)
- [Features](#features)

---

## I/O

### `imread()`

Reads an image from a file.

```python
image = imread("image.jpg")
```

### `imwrite()`

Saves an image to a file.

```python
imwrite(image, "output.jpg")
```

### `imshow()`

Displays an image.

```python
imshow(image)
```

### `imfinfo()`

Returns information about an image file.

```python
info = imfinfo("image.jpg")
```

---

## Color

### `rgb2gray()`

Converts an RGB image to grayscale.

```python
gray = rgb2gray(image)
```

### `rgb2hsv()`

Converts an RGB image to HSV color space.

```python
hsv = rgb2hsv(image)
```

### `hsv2rgb()`

Converts an HSV image to RGB.

```python
rgb = hsv2rgb(hsv)
```

---

## Geometry

### `imresize()`

Resizes an image using a scale factor or target dimensions.

```python
resized = imresize(image, 0.5)
resized = imresize(image, (256, 256))
```

### `imrotate()`

Rotates an image by a specified angle.

```python
rotated = imrotate(image, 90)
```

### `imcrop()`

Crops a selected region from an image.

```python
cropped = imcrop(image, region)
```

### `padarray()`

Adds padding around an image.

```python
padded = padarray(image, padding)
```

---

## Arithmetic

### `imadd()`

Adds two images or an image and a value.

```python
result = imadd(image1, image2)
```

### `imsubtract()`

Subtracts one image from another.

```python
result = imsubtract(image1, image2)
```

### `immultiply()`

Multiplies an image by another image or value.

```python
result = immultiply(image, 2)
```

### `imdivide()`

Divides an image by another image or value.

```python
result = imdivide(image, 2)
```

### `imcomplement()`

Produces the intensity complement of an image.

```python
result = imcomplement(image)
```

### `imabsdiff()`

Calculates the absolute difference between two images.

```python
result = imabsdiff(image1, image2)
```

---

## Histogram

### `imhist()`

Calculates and returns the intensity histogram of an image.

```python
histogram = imhist(image)
```

### `histeq()`

Performs histogram equalization to improve image contrast.

```python
result = histeq(image)
```

### `adapthisteq()`

Performs adaptive histogram equalization to enhance local contrast.

```python
result = adapthisteq(image)
```

---

## Filtering

### `imfilter()`

Applies a filter kernel to an image.

```python
result = imfilter(image, kernel)
```

### `imgaussfilt()`

Applies Gaussian smoothing to an image.

```python
result = imgaussfilt(image, sigma)
```

### `medfilt2()`

Applies a 2D median filter for noise reduction.

```python
result = medfilt2(image, size)
```

---

## Edges

### `edge()`

Detects edges in an image.

```python
edges = edge(image)
```

### `imgradient()`

Calculates the gradient magnitude and direction of an image.

```python
magnitude, direction = imgradient(image)
```

### `imgradientxy()`

Calculates the horizontal and vertical image gradients.

```python
gx, gy = imgradientxy(image)
```

---

## Morphology

### `imerode()`

Shrinks objects in an image using morphological erosion.

```python
result = imerode(image, se)
```

### `imdilate()`

Expands objects in an image using morphological dilation.

```python
result = imdilate(image, se)
```

### `imopen()`

Performs morphological opening.

```python
result = imopen(image, se)
```

### `imclose()`

Performs morphological closing.

```python
result = imclose(image, se)
```

---

## Binary Images

### `imbinarize()`

Converts a grayscale image into a binary image.

```python
binary = imbinarize(image)
```

### `bwareaopen()`

Removes small connected objects from a binary image.

```python
result = bwareaopen(binary, min_size)
```

### `bwperim()`

Finds the boundaries of objects in a binary image.

```python
perimeter = bwperim(binary)
```

### `bwlabel()`

Labels connected components in a binary image.

```python
labels = bwlabel(binary)
```

### `bwconncomp()`

Finds connected components in a binary image.

```python
components = bwconncomp(binary)
```

---

## Measurement

### `regionprops()`

Measures properties of connected regions in an image.

```python
properties = regionprops(labels)
```

Common measurements include:

- `Area`
- `Centroid`
- `BoundingBox`
- `Eccentricity`
- `Extent`
- `MajorAxisLength`
- `MinorAxisLength`
- `Perimeter`
- `Solidity`

---

## Segmentation

### `graythresh()`

Calculates an automatic threshold using Otsu's method.

```python
threshold = graythresh(image)
```

### `adaptthresh()`

Calculates a locally adaptive threshold for an image.

```python
threshold = adaptthresh(image)
```

### `watershed()`

Performs watershed-based image segmentation.

```python
labels = watershed(image, markers)
```

---

## Texture

### `graycomatrix()`

Creates a Gray-Level Co-occurrence Matrix (GLCM) for texture analysis.

```python
glcm = graycomatrix(image)
```

### `graycoprops()`

Calculates texture properties from a GLCM.

```python
properties = graycoprops(glcm, "contrast")
```

Available properties include:

- `contrast`
- `dissimilarity`
- `homogeneity`
- `energy`
- `correlation`
- `ASM`

---

## Transforms

### `fft2()`

Computes the 2D Fast Fourier Transform of an image.

```python
frequency = fft2(image)
```

### `ifft2()`

Computes the inverse 2D Fourier Transform.

```python
image = ifft2(frequency)
```

### `fftshift()`

Shifts the zero-frequency component to the center of the frequency spectrum.

```python
shifted = fftshift(frequency)
```

### `dct2()`

Computes the 2D Discrete Cosine Transform.

```python
coefficients = dct2(image)
```

### `idct2()`

Computes the inverse 2D Discrete Cosine Transform.

```python
image = idct2(coefficients)
```

---

## Noise

### `imnoise()`

Adds noise to an image.

```python
noisy = imnoise(image, "gaussian")
```

Example:

```python
noisy = imnoise(image, "s&p")
```

---

## Features

### `extractHOGFeatures()`

Extracts Histogram of Oriented Gradients (HOG) features from an image.

```python
features = extractHOGFeatures(image)
```

### `detectSURFFeatures()`

Detects SURF feature points in an image.

```python
features = detectSURFFeatures(image)
```

### `detectORBFeatures()`

Detects ORB feature points and descriptors.

```python
features = detectORBFeatures(image)
```
