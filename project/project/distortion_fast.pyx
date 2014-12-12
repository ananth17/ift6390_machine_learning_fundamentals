import heapq
import numpy as np
cimport numpy as np


cdef inline bint richcmp_helper(int compare, int op):
    """Returns True/False for each compare operation given an op code.
    Compare should act similarly to Java's comparable interface"""
    if op == 2: # ==
        return compare == 0
    elif op == 3: # !=
        return compare != 0
    elif op == 0: # <
        return compare < 0
    elif op == 1: # <=
        return compare <= 0
    elif op == 4: # >
        return compare > 0
    elif op == 5: # >=
        return compare >= 0


cdef class PixelDist:
    """object used during optimization"""
    cdef readonly int index, sgn
    cdef readonly double max_dist, abs_gradient
    cdef public double distortion, gradient_cost
    
    
    def __init__(self, int index, double gradient, double pixel):
        self.index = index
        self.sgn = np.sign(gradient)
        self.abs_gradient = np.abs(gradient)
        self.distortion = 0.
        self.max_dist = pixel if self.sgn==-1 else 1 - pixel  # to keep the value between 0 and 1
        self.gradient_cost = self.get_gradient_cost()
    
    def get_gradient_cost(self):
        # we put the result in negative to
        # make the min heap a max heap (on the real value)
        return -(self.abs_gradient /(self.incrementation_cost()))

    def can_update(self):
        # checks if still possible to add distortion
        return self.distortion < self.max_dist

    def add_distortion(self):
        self.distortion += 1./256.
        self.gradient_cost = self.get_gradient_cost()
        return 1./256. * self.abs_gradient

    def incrementation_cost(self):
        cdef double low = self.distortion * self.distortion
        cdef double high = (self.distortion + 1./256.) * (self.distortion + 1./256.)
        return(high - low)
    
    def __richcmp__(PixelDist self, PixelDist other not None, int op):
        cdef int compare
        cdef double v1 = self.gradient_cost
        cdef double v2 = other.gradient_cost
        if v1 > v2:
            compare = 1
        elif v1 < v2:
            compare = -1
        else:
            compare = 0
        return richcmp_helper(compare, op)
     
    def __str__(self):
        return str(self.index)+ "; " + str(self.abs_gradient) + "; " + str(self.distortion)


cpdef distort(np.ndarray digit, int original_class, int target, np.ndarray coefs, np.ndarray scores, double best_dist_yet):
    # greedy optimization procedure for finding squared norm to closest image
    # similar to knapsack
    
    # smallest value to add
    cdef double quantum = 1./256.
    # difference between the coefficients of the class equations
    cdef np.ndarray difference = coefs[target] - coefs[original_class]
    # absolute difference of the class equations
    cdef np.ndarray abs_difference = np.abs(difference)
    # sign of the difference
    cdef  np.ndarray sgn = np.sign(difference)
    # maximum values for distortion
    cdef np.ndarray boundaries = np.zeros(len(digit))
    cdef double x,y
    for i in range(0, len(digit)):
        x = digit[i]
        y = sgn[i]
        if y == -1:
            boundaries[i] = x
        else:
            boundaries[i] = 1 - x
    
    # value to go over
    cdef double threshold = scores[original_class] - scores[target]
    assert(threshold > 0.), "the digit is already misclassified"
    
    # values to keep track of for optimization
    cdef double current_score = 0.
    cdef double distance = 0.
    
    # heap structure for fast maximum
    heap = [PixelDist(i, difference[i], digit[i]) for i in range(0, len(difference))]
    heapq.heapify(heap)
    cdef PixelDist best
    
    # greedy optimization procedure
    while (current_score < threshold):
        best = heapq.heappop(heap)

        # update and add to heap
        if best.can_update():
            distance += best.incrementation_cost()
            if distance > best_dist_yet:  # stop the optimization if for nothing
                return float('inf')
            current_score += best.add_distortion()
            heapq.heappush(heap, best)
    return distance



cpdef get_distance(digit, original_class, classifier):
    # get closest squared norm
    scores = classifier.decision_function(digit)[0]
    coefficients = classifier.coef_
    smallest = float('inf')
    for i in range(0, 10):
        if i!=original_class:
            distance = distort(digit, original_class, i, coefficients, scores, smallest)
            if smallest > distance:
                smallest = distance
    return smallest



cpdef get_distance_and_class(digit, original_class, classifier):
    # get closest squared norm
    scores = classifier.decision_function(digit)[0]
    coefficients = classifier.coef_
    smallest = (original_class, float('inf'))
    for i in range(0, 10):
        if i!=original_class:
            distance = (i, distort(digit, original_class, i, coefficients, scores, smallest[1]))
            if smallest[1] > distance[1]:
                smallest = distance
    return smallest
    
