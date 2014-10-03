"""sample scripts for covariance related stuff"""


def cov(v1, v2):
    assert len(v1) == len(v2), "vectors should be of same size"
    m1 = sum(v1) / (len(v1))
    m2 = sum(v2) / (len(v2))
    cov = 0
    for i in range(0, len(v1)):
        cov += (v1[i] - m1)*(v2[i]-m2)
    return cov / (len(v1) -1 )


def cov_matrix(M):
  """calculates the covariance matrix from the matrix,
     each line is a group of observations"""
  ret = []
  for i in range(0, len(M)):
    line = []
    for j in range(0, len(M)):
      line.append(cov(M[i], M[j]))
    ret.append(line)
  return ret


# e.g.

arr = [[ 4.  ,  4.2 ,  3.9 ,  4.3 ,  4.1 ],
       [ 2.  ,  2.1 ,  2.  ,  2.1 ,  2.2 ],
       [ 0.6 ,  0.59,  0.58,  0.62,  0.63]]

covM = cov_matrix(arr)
