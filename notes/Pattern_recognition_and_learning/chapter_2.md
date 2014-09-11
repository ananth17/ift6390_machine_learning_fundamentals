# Chapter 2: probability distributions


## overview


Role of the distributions is to model the probability distribution p(x) of
a random variable x, given a finite set x1, .., xn of observations.


**Parametric distributions**: governed by a small number of adaptive parameters, such as the mean and variance in
the case of a Gaussian. These distributions are examples of the exponential family of distributions.

- discrete: binomial and multinomial distributions
- continuous: Gaussian distribution

- need a procedure for determining suitable values for the parameters, given an observed data set.

**frequentist approach**: choose specific values forparameters
by optimizing some criterion, such as the likelihood function.

**bayesian approach**: introduce prior distributions over the parameters and use
Bayes’ theorem to compute the corresponding posterior distribution given the observed data.


Parametric approach assumes a specific functional form for the distribution,
which may turn out to be inappropriate for a particular application.


Alternative approach is given by nonparametric density estimation methods in which
the form of the distribution typically depends on the size of the data
set. They still contain parameters, but they control the model complexity
rather than the form of the distribution.


**likelihood function**: probability of the observed outcome given the parameter value


## binary variables


Single binary random variable x ∈ {0, 1}.
For example, outcome of flipping a coin.
Head = 1, tail = 0.
The probability of x = 1 will be denoted by the parameter μ.
p(x = 1|μ) = μ; p(x = 0|μ) = 1 − μ.


### bernouilli distribution


**Bern**(x|μ) = (μ^x) * ((1 − μ)^(1 − x))

**mean** E[x] = μ
**variance** var[x] = μ(1 − μ)
**likelihood function** p(D|μ) = mult{1..N} p(xn|μ) = mult{1..N} μ^(xn) * (1 - μ)^(1-xn)



~~~python

    from random import uniform
    from math import log


    class bernouilli(object):
      """ P(1) = μ, P(0) = 1 - μ"""

      def __init__(self, mu):
        assert isinstance(mu, float) and 0 <= mu <= 1
        self.mu = mu  # probability of 1

      def rand(self):
        """returns a random observation of bernouilli distribution"""
        if uniform(0, 1) < self.mu:
          return 1
        else:
          return 0


    def bernouilli_likelihood(D, mu):
      """likelihood probability of the observations according to mu"""
      probability = 1
      for realization in D:
        if realization == 1:
          probability *= mu
        else:
          probability *= (1 - mu)
      return probability


    def bernouilli_log_likelihood(D, mu):
      """log likelihood probability"""
      return log(bernouilli_likelihood(D, mu))


    def average_sample_bernouilli(D):
      return sum(D) / len(D)


    distribution = bernouilli(0.5)
    print("[")
    for _ in range(1, 10):
      print(distribution.rand())
    print("]")

~~~


## binomial distribution


both variance and averge calculation can give severly over-fitted results

~~~python

    def factorial(n):
      assert n >= 0 and isinstance(n, int), "wrong input"
      ret = 1
      if n == 0:
        return ret
      for i in range(1, n+1):
        ret *= i
      return ret

    def combination(N, m):
      return factorial(N) // ((factorial(N - m)) * factorial(m))

    def binomial_likelihood(D, mu):
      """estimate the probability of getting m heads out of N"""
      m = sum(D)
      N = len(D)
      return combination(N, m) * (mu ** m) * ((1 - mu) ** (N - m))

~~~


## beta distribution




## gaussian distribution




















