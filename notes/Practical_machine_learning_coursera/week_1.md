
# week 1


## example of detecting spam

question -> input data -> features -> algorithm -> parameters -> evaluation

classify email in SPAM / HAM

labeled emails

text of the email, address

algorithm of choice

parameters of choice

evaluate the performance as seen applicable (in this case, accuracy of classification)


## relative importance of each step

question -> input data -> features -> algorithms


### Question


- question is the most important part (getting it right is crucial)
- data usually is never as good as needed
- features extraction is usually hard
- algorithm usually is the least important part of the process


Quoted from Tukey (FFT and other):
**The combination of some data and an aching desire for an answer does not ensure that a reasonable answer can be extracted from any given body of data**


An important part is to know when to give up.


### Data: Garbage in = garbage out


The value of available information is hard to assess.

Often, getting more data beats better statistics.


### Features


ideal features

- lead to data compression
- retain relevant information
- created based on expert application knowledge


common mistakes

- trying to automate feature selection (doesn't allow understanding)
    - neural network tanks vs apc and day vs night
    - deep learning allows automation of feature selection (google's cat classifier)
- not paying attention to data-specific quirks
- throwing away information unnecessarily



### Algorithm: matters less than you think


!["Refer to this article"](../../documents/coursera/algorithms_matter_less.pdf)



### Issues to consider


- interpretable: so that people can understand it
    - easy to explain therefore
- accuracy vs other qualities
- scalable (easy to train and test)
- fast



### Prediction is about accuracy tradeoff

- interpretability matters
    - people like decision trees (and ones that make sense too)
    - google flu example, hard to say where the problem is
- scalability matters
    - netflix challenge winner's algorithm was never implemented because of scalability issues



## In sample and out of sample error


### In sample vs out of sample

**in sample error**: error rate you get on the same data set used to build the predictor (resubstitution error)

**out of sample error**: error rate on a new data set. Sometimes referred to as **generalization error**.


key ideas:

- out of sample error is what you care about
- in sample error < out of sample error
- the reason is overfitting
    - matching your algorithm to the data you have (too much, no generalization possibilities)


### Overfitting


Data has 2 parts: signal and noise.
A perfect in-sample predictor usually captures both signal and noise and won't generalize well.



## Prediction study design


### Building steps


1. Define error rate
2. Split data into training, testing and validation (optional)
3. During training, pick features (use cross-validation)
4. On the training set pick prediction function (use cross-validation)
5. If no validation, apply once to the test set
6. If validation is available, apply to the test set and refine and apply once to the validation set

The idea is to hold out some data to later evaluate performance (generalization). Always.


### Avoid small sample sizes

- could easily get estimate that is way off due to random sampling


### rules of thumb for prediction study design


**large sample size**

- 60% training
- 20% test
- 20% validation


**medium sample size**

- 60% training
- 40% test


**small sample size**

- cross-validation
- report caveat of small sample size



### some principles to remember

- set test/validation set aside and don't look at it
- randomly sample training and test
- data sets must reflect structure of the problem
    - if prediction evolve with time, split train/test in time chuncks
    - this is referred to as backtesting in finance
- all subsets should reflect as much diversity as possible
    - random assignment does this
    - can try to balance by feature, but trickier



## Types of errors


### basic terms


**TP**: correctly identified
**FP**: incorrectly identified
**TN**: correctly rejected
**FN**: incorrectly rejected


**sensitivity**: Pr(positive test | disease)
**specificity**: Pr(negative test | no disease)
**Positive predictive value**: Pr(disease | positive test)
**Negative predictive value**: Pr(no disease | negative test)
**accuracy**: (TP + TN) / (TP + FP + FN + TN)


### screening tests example

Test: 99% specificity and sensitivity
Disease: 0.1% prevalence

TP: 99
TN: 96901
FP: 999
FN: 1



### error estimation


for continuous data, the goal is to see how close we are to the truth

MSE = (1/n) * sum((prediction - truth)^2)
RMSE = sqrt(MSE)



### common error measures

1. MSE or RMSE
2. Median absolute deviation: continuous data, often more robust
3. Sensitivity (recall): if you want few missed positives
4. Specificity: if you want few negatives labeled positives
5. Accuracy: (TP + TN) / (TP + TN + FP + FN), trueness over total
6. concordance



## ROC, receiver operating characteristic


- allows to see wether or not an algorithm is performing
- the bigger the area, the better
- AUC (area under curve): 0.5 -> equivalent to random guessing, less than that, really shitty
- usually AUC > 0.8, generally considered good
- farther you are on left high, the better



## Cross-validation













