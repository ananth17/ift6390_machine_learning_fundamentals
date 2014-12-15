import math
import numpy
import matplotlib.pyplot as plt
import seaborn

def confusion_matrix(data):
    """ """
    f = plt.figure(figsize=(10, 8))
    seaborn.heatmap(data, annot=True, cmap='Blues')
    plt.xlabel("predicted class")
    plt.ylabel("true class")
    plt.show()
    return f


def show_mnist_digit(digit):
    """displays mnist digit with scaling (outside 0 and 1)"""
    low = numpy.min(digit)
    high = numpy.min(digit)
    if high < 1.:
        high = 1.
    if low > 0.:
        low = 0.
    plt.imshow(numpy.reshape(digit,(28,28)), cmap=plt.cm.gray, interpolation='none', vmin=low, vmax=high)
    plt.colorbar()
    plt.show()


def assert_legit_mnist(image):
    """used to verify that the digit is 8-bit grayscale, between 0 and 1"""
    low = numpy.min(digit)
    high = numpy.max(digit)
    if low < 0. or high > 1.:
        return False
    integers = numpy.equal(numpy.mod(256. * image, 1.), 0.)
    return numpy.sum(integers) == 784.



