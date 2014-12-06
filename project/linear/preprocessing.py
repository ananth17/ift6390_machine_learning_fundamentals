# preprocessing functions

# - deskewing
# - ink normalization

from scipy.spatial import distance
from scipy.ndimage import interpolation
import numpy


def ink_normalize(x):
    """ ink normalization for image"""
    return x /numpy.linalg.norm(x)


def moments ( image ) :
    """ helper for deskewing image"""
    # taken from
    # http://nbviewer.ipython.org/github/tmbdev/teaching-lw/blob/master/102-nearest-neighbor-and-preproc.ipynb
    c0,c1 = numpy.mgrid[:image.shape[0],:image.shape[1]]
    imgsum = numpy.sum(image)
    m0 = numpy.sum(c0*image )/imgsum
    m1 = numpy.sum(c1*image )/imgsum
    m00 = numpy.sum(( c0 - m0 )*( c0 - m0 )*image)/ imgsum
    m11 = numpy.sum(( c1 - m1 )*( c1 - m1 )*image)/ imgsum
    m01 = numpy.sum(( c0 - m0 )*( c1 - m1 )*image)/ imgsum
    return numpy.array([ m0, m1]), numpy.array([[m00, m01], [m01, m11]])


def deskew(image):
    """deskew image"""
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = numpy.array([[1,0],[alpha,1]])
    ocenter = numpy.array(image.shape)/2.0
    offset = c-numpy.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)


def to8bit(image):
    """return an 8-bit version of the image by centering
    (between 0 and 1) and round to closest 1/256 fraction"""
    low  = numpy.min(image)
    high = numpy.max(image)
    interval = high - low
    # perform centering (between 0 and 1)
    new_image = (image - low) / interval
    
    # perform 8bit conversion (only multiples of 1/256)
    new_8bit_image = numpy.ceil(new_image * 256.)/256.
    return new_8bit_image

