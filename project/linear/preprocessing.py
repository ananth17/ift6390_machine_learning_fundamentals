# preprocessing functions

# - deskewing
# - ink normalization

from scipy.spatial import distance
from scipy.ndimage import interpolation


def ink_normalize(x):
    """ ink normalization for image"""
    return x /np.linalg.norm(x)

def moments ( image ) :
    """ helper for deskewing image"""
    # taken from
    # http://nbviewer.ipython.org/github/tmbdev/teaching-lw/blob/master/102-nearest-neighbor-and-preproc.ipynb
    c0,c1 = mgrid[:image.shape[0],:image.shape[1]]
    imgsum = np.sum(image)
    m0 = np.sum(c0*image )/imgsum
    m1 = np.sum(c1*image )/imgsum
    m00 = np.sum(( c0 - m0 )*( c0 - m0 )*image)/ imgsum
    m11 = np.sum(( c1 - m1 )*( c1 - m1 )*image)/ imgsum
    m01 = np.sum(( c0 - m0 )*( c1 - m1 )*image)/ imgsum
    return array([ m0, m1]), np.array([[m00, m01], [m01, m11]])


def deskew(image):
    """deskew image"""
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = array([[1,0],[alpha,1]])
    ocenter = array(image.shape)/2.0
    offset = c-dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)

