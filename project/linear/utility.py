import numpy
import matplotlib.pyplot as plt


def confusion_matrix(classification, name):
    """ """
    data = numpy.empty((0, 10))
    for i in range(0, 10):
        row = numpy.zeros(10)
        for j in range(0, 10):
            row[j] = classification[i][j]
        row /= float(numpy.sum(row))
        data = numpy.vstack((data, row))
    plt.figure(figsize = (10, 10))
    plt.title(name)
    plt.ylabel("image class")
    plt.xlabel("predicted class")
    plt.xticks(range(0,10))
    plt.yticks(range(0,10))
    plt.imshow(data, cmap=plt.cm.summer_r, interpolation='none')
    plt.colorbar()
    plt.show()
    class_accuracy = []
    for i in range(0, 10):
        class_accuracy.append(data[i][i])
    plt.figure(figsize = (9, 7))
    plt.bar(range(0,10), class_accuracy)
    plt.title('accuracy per class')
    plt.xticks(range(0,10))
    plt.show()
    return data


def show_mnist_digit(digit):
    """displays mnist digit with scaling (outside 0 and 1)"""
    low = numpy.min(digit)
    high = numpy.min(digit)
    if high < 1.:
        high = 1
    if low > 0.:
        low = 0
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


