""" general statistics functions for analysis of classifier performance"""

import numpy
import seaborn

def get_global_accuracy(data, target, classifier):
    """ returns the global accuracy of the classifier on data and target"""
    return numpy.sum(numpy.equal(classifier.predict(data), target)) / float(len(target))



def get_class_performance(train_x, train_y, classifier):
    """ returns the class-based performance of the classifier"""
    # per-class misclassification rate
    # per-class misclassification identities
    result = numpy.zeros((10, 10))
    predicted = classifier.predict(train_x)
    
    for i in range(0, len(predicted)):
        result[train_y[i]][predicted[i]] += 1
    
    return result/result.sum(axis=1)[:,None]

#def get_class_performance_common(train_x, train_y, classifier1, classifier2):
    #result = numpy.zeros((10, 10))
    #predicted1 = classifier1.predict(train_x)
    #predicted2 = classifier2.predict(train_x)
    #for i in range(0, len(predicted1)):
      