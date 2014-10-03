""" this shows the logsumexp trick"""

import math

def logsumexp1(vector):
  s = 0
  for i in vector:
    s += math.exp(i)
  return math.log(s)


def logsumexp2(vector):
  s = 0
  A = -1 * max(vector)
  for i in vector:
     s += math.exp(i + A)
  return math.log(s) - A


print(logsumexp1([1,2,3,4,5]))

print(logsumexp2([1,2,3,4,5]))
