

# Week 3


## predicting with trees


### key ideas

- iteratively split variables into groups
- evaluate homogeneity within each group
- split again if necessary


pros:

- easy to interpret
- better performance in nonlinear settings

cons:

- without pruning/cross-validation, can lead to overfitting
- harder to estimate uncertainty
- results may be variable


### example of decision tree

obama-clinton divide, built a prediction model that asked for properties of the county
(>20% african-american?) -> Obama


in leafs of the tree, the predictions were mostly homogenous (split into most homogenous
groups askin the least possible questions)


### basic algorithm

1. start with all variables in one group
2. find the variable that best separates the classes
3. divide the data into two groups: ("leaves") on that split ("node")
4. within each split, reapply the steps 2 and 3
5. continue until the groups are too small or sufficiently homogenous


### measures of impurity

- misclassification error
    - 0 = perfect purity, 0.5 = no purity

- gini index
    - 0 = perfect purity, 0.5 = no purity


### example: iris data

can make a dendogram of decisions that will explicitely separate the classes


### notes and further resources

- nonlinear models
    - use interactions between variables
    - data transformation can be less important (monotone transformations change nothing)
    - trees can also be used for regression problems (continuous outcome)
- multiple tree building options in R



## Bagging (bootstrap + aggregating)


### basic idea


1. resample cases and recalculate predictions
2. average or majority vote

- similar bias
- reduced variance
- most useful for non-linear functions


### ozone data


~~~R

    library(ElemStatLear); data(ozone, package="ElemStatLearn")
    ozone <- ozone[order(ozone$ozone),]
    
    
    
    
~~~

#### bagged loess (locally weighted sctterplot smoothing)



### notes and further resources

- bagging is most useful for nonlinear models
- often used with trees, an extension is random forests
- several models use bagging in caret's train function



## Random forests

1. bootstrap samples
2. at each split, bootstrap variables
3. grow multiple trees and vote

pros:

1. accuracy (one of the best, in Kaggle at least)

cons:

1. speed
2. interpretability
3. overfitting

idea is to either average (continuous outcome I suppose) or use democracy :) on the output of the
multiple trees




















