import numpy as np

# load data
train_set = np.loadtxt("train_images.txt", delimiter=",")
train_labels = np.loadtxt("train_labels.txt", delimiter=",")

test_set = np.loadtxt("test_images.txt", delimiter=",")
test_labels = np.loadtxt("test_labels.txt", delimiter=",")


class BernouilliNaiveBayes(object):
    """Naive Bayes Classifier"""
    
    def __init__(self):
        self.priors = np.array([])
        self.classConditionals = np.array([])
    
    def train(self, x, y):
        """We use Laplace smoothing (add-one) in the training of the prior and likelihood"""
        # posterior = prior * likelihood / evicende
        # prior
        number_of_observations = np.sum(y, axis = 0)
        self.priors = number_of_observations / float(y.shape[0])

        # likelihood
        self.classConditionals = np.array([np.zeros(x.shape[1]) for _ in range(0, y.shape[1])])
        for i in range(0, x.shape[0]):
            category = np.argmax(y[i])
            self.classConditionals[category] = np.add(self.classConditionals[category], x[i])
        # laplace smoothing
        number_of_observations = np.add(number_of_observations, 2)
        for i in range(0, y.shape[1]):
            self.classConditionals[i] = np.divide(np.add(self.classConditionals[i], 1), number_of_observations[i])
        return
    
    
    def calculate_log_likelihood(self, classIndex, image):
        """calculate the probability of the image given the class"""
        assert 0 <= classIndex < self.classConditionals.shape[0]
        prob = np.zeros(len(image))
        learned = self.classConditionals[classIndex]
        for i in range(0, len(learned)):
            if(image[i] == 1.0):
                prob[i] = learned[i]
            elif(image[i] == 0.0):
                prob[i] = 1.0 - learned[i]
            else:
              print("Major error")
        return np.sum(np.log(prob))
    
    def calculate_log_posterior(self, classIndex, image):
        """calculate the posterior probability"""
        # posterior = prior * likelihood
        logL = self.calculate_log_likelihood(classIndex, image)
        prior = self.priors[classIndex]
        log_posterior = logL + np.log(prior)
        return log_posterior
    
    def findClass(self, x):
        """ """
        
        return np.argmax([self.calculate_log_posterior(i, x) for i in range(0, self.classConditionals.shape[0])])
        
    def showLikelihoods(self):
        for i in range(0, self.classConditionals.shape[0]):
            show(self.classConditionals[i,:])
            
    def compute_predictions(self, data_set):
        """return the maximum a priori log likelihood"""
        # compute the log likelihood for a single x
        predictions = []
        for i in range(0, data_set.shape[0]):
            predictions.append(self.findClass(data_set[i,:]))
        return predictions
        


    def check_fit(self, x, y):
        number_of_examples = x.shape[0]
        number_of_successful = np.sum(self.compute_predictions(x) == np.argmax(y, axis=1))
        
        return (number_of_successful, number_of_examples, float(number_of_successful)/number_of_examples)
      
      
      
# testing
classifier = BernouilliNaiveBayes()
classifier.train(train_set, train_labels)