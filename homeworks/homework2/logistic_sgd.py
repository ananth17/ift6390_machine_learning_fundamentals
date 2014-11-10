

import numpy
import theano
import theano.tensor as T


class LogisticRegression(object):
    """Output layer with softmax activation"""

    def __init__(self, input, n_in, n_out):
        """
        :param input: symbolic variable that describes the input
         of the architecture (one minibatch)

        :param n_in: number of input units, the dimension of
         the space in which the datapoints lie

        :param n_out: number of output units, the dimension of
         the space in which the labels lie
        """

        # zero initialized W
        # dim(W) = (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        # zero initialized b
        # dim(b) = (n_out)
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # softmax activation function
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # predicted function (max softmax)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # all the parameters that need to be fitted
        self.params = [self.W, self.b]


    def negative_log_likelihood(self, y):
        """
        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data():
    '''
    Loads the data
    '''

    test_x = numpy.loadtxt('test_images.txt', delimiter=',')
    test_y = numpy.argmax(numpy.loadtxt('test_labels.txt', delimiter=','), axis=1)
    train_x = numpy.loadtxt('train_images.txt', delimiter=',')
    train_y = numpy.argmax(numpy.loadtxt('train_labels.txt', delimiter=','), axis=1)


    def shared_dataset(data_x, data_y, borrow=True):
        """
        Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_x, test_y = shared_dataset(test_x, test_y)
    train_x, train_y = shared_dataset(train_x, train_y)

    rval = [(train_x, train_y),
            (test_x, test_y)]
    return rval
