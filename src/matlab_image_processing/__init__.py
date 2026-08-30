"""MATLAB-style image processing functions for Python."""

__version__ = "0.1.0"

from .io import imread, imwrite, imshow, imfinfo
from .color import rgb2gray, rgb2hsv, hsv2rgb
from .geometry import imresize, imrotate, imcrop, padarray
from .arithmetic import (
    imadd,
    imsubtract,
    immultiply,
    imdivide,
    imcomplement,
    imabsdiff,
)
from .intensity import imadjust, rescale, stretchlim
from .histogram import imhist, histeq, adapthisteq
from .filtering import imfilter, imgaussfilt, medfilt2
from .edges import edge, imgradient, imgradientxy
from .morphology import imerode, imdilate, imopen, imclose
from .binary import imbinarize, bwareaopen, bwperim, bwlabel, bwconncomp
from .measurement import regionprops
from .segmentation import graythresh, adaptthresh, watershed
from .texture import graycomatrix, graycoprops
from .transforms import fft2, ifft2, fftshift, dct2, idct2
from .noise import imnoise
from .features import (
    extractHOGFeatures,
    detectSURFFeatures,
    detectORBFeatures,
)