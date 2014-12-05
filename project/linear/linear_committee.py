"""committee of linear classifiers"""

import collections
import numpy
import sklearn


class LinearCommittee(object):
    """ committee of linear classifiers (SGDclassifier)"""

    def __init__(self, numOfVoters, policy, loss_function):
        """
        numOfVoters: number of linear classifiers in the committee
        policy: one of [majority, average, median]
        """
        assert(numOfVoters > 0), "number of voters doesn't make sense"
        assert(policy in ['majority', 'average', 'median']), "invalid policy"
        # initialize the classifiers, untrained
        self._policy = policy
        self._numOfVoters = numOfVoters
        if policy in ('median', 'average'):
            assert loss_function in ('log'), "the loss function doesn't allow probability prediction"
        self._classifiers = [sklearn.linear_model.SGDClassifier(loss=loss_function) for _ in range(0, numOfVoters)]

    def fit(self, index, X, Y):
        """ fit one of the classifier to given data, useful because
            we may use different preprocessing methods on the data"""
        assert(index in xrange(0, self._numOfVoters)), "invalid index"
        self._classifiers[index].fit(X,Y)

    def getClassifier(self, index):
        """ access classifier at index"""
        assert(index in xrange(0, self._numOfVoters)), "invalid index"
        return self._classifiers[index]
    
    def vote(self, X):
        """ vote on the class, given the data, based on the policy
            works only on a single input for now (use map)"""
        # preprocess the input with the preprocessing functions
        assert(len(X) == self._numOfVoters), "missing some preprocessed images"
        pairs = zip(self._classifiers, X)
        if self._policy=='majority':
            # majority vote
            votes = map(lambda (classifier, image): classifier.predict(image)[0], pairs)
            vote_count = collections.Counter(votes)
            #print vote_count
            return vote_count.most_common()[0][0]
        #elif self._policy in ('average', 'median'):
            ## average the class probabilities for the n classifiers and choose
            ## the class with the highest average posterior class probability
            #probabilities = numpy.array(self._classifiers[0].predict_proba(X))
            #for i in xrange(1, self._numOfVoters):
              #probabilities = np.vstack(probabilities, self._classifies[i].predict_proba(X))
            ## compute either the median or the average
            #if self._policy == 'average':
              #return np.argmax(np.average(probabilities, axis=0))
            #else:
              #return np.argmax(np.median(probabilities, axis=0))




