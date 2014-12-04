""" general statistics functions for analysis of classifier performance"""

import numpy


def get_global_accuracy(data, target, classifier):
    """ returns the global accuracy of the classifier on data and target"""
    return numpy.sum(numpy.equal(classifier.predict(data), target)) / float(len(target))



def get_class_based_performance(class_to_data, classifier):
    """ returns the class-based performance of the classifier"""
    # per-class misclassification rate
    # per-class misclassification identities
    class_accuracy = dict()
    class_destination = dict()
    
    for (c, data) in class_to_data.iteritems():
        predicted = classifier.predict(data)
        predictions_length = float(len(predicted))
        
        class_accuracy[c] = numpy.sum(map(lambda x:x==c, predicted)) / predictions_length
        
        class_destination[c] = dict()
        
        # initialize the destinations with 0
        for d in class_to_data.keys():
            class_destination[c][d] = 0
        
        for predicted_class in predicted:
            class_destination[c][predicted_class] += 1

    return (class_accuracy, class_destination)

