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








## gaussian distribution




