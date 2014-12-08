""" distortion methods for mnist digit adversary examples generation"""
import numpy
import sklearn
import heapq


def distort_random(digit, original_class, target, classifier, seed=42, epsilon=0.1, hard_threshold = 100000):
    """ distort random position in the image, in the optimal direction (using gradient information)"""
    assert(isinstance(classifier, sklearn.linear_model.stochastic_gradient.SGDClassifier))
    assert(epsilon > 0)

    numpy.random.seed(seed)
    quantum = 1./256.  # we need the image to stay in 8-bit grayscale, otherwise it's cheating
    
    distortion = numpy.zeros(len(digit))
    difference = classifier.coef_[target] - classifier.coef_[original_class]
    
    abs_difference = numpy.abs(difference)
    sgn = numpy.sign(difference)
    
    boundaries = numpy.array(map(lambda (x,y): x if y==-1 else 1-x, zip(digit, sgn)))
    
    # get the initial scores to know the value to go over
    scores = classifier.decision_function(digit)[0]
    threshold = scores[original_class] - scores[target]
    
    current_score = 0.
    number_modifications = 0
    
    assert(numpy.argmax(scores) == original_class)
    while (current_score < threshold)  and (number_modifications < hard_threshold):
        index = numpy.random.randint(low=0, high=len(digit))
        
        # check if the value should be skipped
        if (abs_difference[index] > epsilon) and (distortion[index] < boundaries[index]):
            gradient = numpy.sign(difference[index])
            distortion[index] += quantum
            # update the counters
            current_score += abs_difference[index] * quantum
            #print "Current score = " + str(current_score)
            number_modifications += 1

    #print number_modifications
    #print current_score
    #print "remaining = " + str(threshold - current_score)
    return distortion*sgn



class Index2Gradient(object):
    def __init__(self, index, gradient, pixel):
        self.index = index
        self.sgn = numpy.sign(gradient)
        self.abs_gradient = numpy.abs(gradient)
        self.distortion = 0.
        self.max_dist = pixel if self.sgn==-1 else 1 - pixel  # to keep the value between 0 and 1
        self.gradient_cost = self.get_gradient_cost()
    
    def get_gradient_cost(self):
        # we put the result in negative to
        # make the min heap a max heap (on the real value)
        low = self.distortion * self.distortion
        high = (self.distortion + 1./256.) * (self.distortion + 1./256.)
        return -(self.abs_gradient /(high - low))

    def can_update(self):
        # checks if still possible to add distortion
        return self.distortion < self.max_dist

    def add_distortion(self):
        self.distortion += 1./256.
        self.gradient_cost = self.get_gradient_cost()
        return 1./256. * self.abs_gradient
    
    def __lt__(self, other):
        return self.gradient_cost < other.gradient_cost
    
    def __le__(self, other):
        return self.gradient_cost <= other.gradient_cost
    
    def __gt__(self, other):
        return not self.__le__(other)
    
    def __ge__(self, other):
        return not self.__lt__(other)
    
    def __str__(self):
        ret = "(i "+str(self.index) + ",d " + str(self.distortion) +",g " +\
        str(self.abs_gradient) + ",c " + str(self.gradient_cost) + ")"
        return ret
    
    def __repr__(self):
        return self.__str__()



def distort_opt_ratio(digit, original_class, target, classifier, get_image=True):
    """uses the best improvement/cost ratio to choose which pixel to update"""
    assert(isinstance(classifier, sklearn.linear_model.stochastic_gradient.SGDClassifier))
    quantum = 1./256.  # we need the image to stay in 8-bit grayscale, otherwise it's cheating
    
    distortion = numpy.zeros(len(digit))
    difference = classifier.coef_[target] - classifier.coef_[original_class]
    
    abs_difference = numpy.abs(difference)
    sgn = numpy.sign(difference)
    
    boundaries = numpy.array(map(lambda (x,y): x if y==-1 else 1-x, zip(digit, sgn)))
    
    # get the initial scores to know the value to go over
    scores = classifier.decision_function(digit)[0]
    threshold = scores[original_class] - scores[target]
    
    current_score = 0.
    number_modifications = 0
    
    scores = classifier.decision_function(digit)[0]
    threshold = scores[original_class] - scores[target]
    full = []
    heap = [Index2Gradient(i, difference[i], digit[i]) for i in range(0, len(difference))]
    heapq.heapify(heap)
    curr_iter = 0
    best = None
    assert(threshold > 0.), "the digit is already misclassified"
    while (current_score < threshold):
        best = heapq.heappop(heap)
        curr_iter += 1
        
        if not best.can_update():
            full.append(best)
        else:
            current_score += best.add_distortion()
            heapq.heappush(heap, best)
            

    if get_image:
      for i in full:
          heap.append(i)
      ret = numpy.zeros(len(digit))
      for dist in heap:
        ret[dist.index] = dist.sgn*dist.distortion
      #print "attained threshold " + str(current_score)
      return ret
    else:
      distance_high = 0.
      for i in heap:
          distance_high += i.distortion * i.distortion
      for i in full:
          distance_high += i.distortion * i.distortion
      d = best.distortion
      distance_low = distance_high - ((d*d) -((d-1./256.) * (d-1./256.)))
      return (distance_low, distance_high)


def get_squared_norm_adversarial(digit, original_class, classifier):
  smallest = (0, float('inf'))
  for i in range(0, 10):
    if i!= original_class:
      dist = distort_opt_ratio(digit, original_class, i, classifier, get_image=False)
      if smallest[1] > dist[1]:
        smallest=dist
  return smallest
