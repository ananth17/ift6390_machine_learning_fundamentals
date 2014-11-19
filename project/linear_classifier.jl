
# class LinearClassifier(object):
#     def __init__(self, shape):
#         self.shape = shape
#         self.W = np.zeros((shape[0], shape[1]))
#
#     def train(self, lmb, x, y):
#         """train the weights on the data, given the regularization parameter
#            using ridge regression
#            inspired by mlpy ridge regression's code
#            https://github.com/sauliusl/mlpy/blob/master/mlpy/ridge.py"""
#
#         #(X'X + lambda I)^-1 X'Y
#         #add the bias term to x
#         x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
#         left = np.dot(x.T, x)
#         special_id = np.identity(x.shape[1])
#         special_id[0][0] = 0.0
#         left = left + (lmb * special_id)
#         left = np.linalg.pinv(left)
#         right = np.dot(x.T, y)
#         self.W = np.dot(left, right)
#
#     def compute_predictions(self, test_data):
#         added_bias = np.append(np.ones((test_data.shape[0], 1)), test_data, axis=1)
#         return np.argmax(np.dot(added_bias, self.W), axis=1)
#
#     def check_fit(self, x, y):
#         number_of_examples = x.shape[0]
#         number_of_successful = np.sum(self.compute_predictions(x) == np.argmax(y, axis=1))
#
#         return (number_of_successful, number_of_examples, float(number_of_successful)/number_of_examples)


type LinearClassifier
  W::Matrix{FloatingPoint}
  number_of_classes::Integer
  input_size::Integer
  
  function LinearClassifier()
    # initialize the linear classifier
    assert (number_of_classes > 0)
    assert (input_size > 0)
    W = zeros(number_of_classes, input_size)
    classifier = new()
    classifier.W = W
    classifier.number_of_classes = number_of_classes
    classifier.input_size = input_size
    clasifier
  end
end

#         #(X'X + lambda I)^-1 X'Y
#         #add the bias term to x
#         x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
#         left = np.dot(x.T, x)
#         special_id = np.identity(x.shape[1])
#         special_id[0][0] = 0.0
#         left = left + (lmb * special_id)
#         left = np.linalg.pinv(left)
#         right = np.dot(x.T, y)
#         self.W = np.dot(left, right)


function train(classifier::LinearClassifier, penalty::FloatingPoint,
               X::Matrix{FloatingPoint}, Y::Vector{Int})
  #(X'X + lambda I)^-1 X'Y
  (X'X + penalty) X'Y
end


function check_fit(x::Matrix{FloatingPoint}, y::Vector{Int})
  #
  
end
