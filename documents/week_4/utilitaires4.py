# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def fonction_1():
    """
    """
    x = np.linspace(0, 100, 250)
    y = x + np.random.normal(loc=0.0,scale=5.0,size=x.shape)

    x -= x.min()
    x /= x.max()

    y -= y.min()
    y /= y.max()

    return x, y

def fonction_2():
    x = np.linspace(-10, 10, 250)
    y = x-x**2+x**3+np.random.normal(loc=0.0,scale=100.,size=x.shape)+2

    x -= x.min()
    x /= x.max()

    y -= y.min()
    y /= y.max()

    return x, y

def plot(x, y, y_hat):
    """
    """
    
    plt.clf()
    plt.scatter(x, y)
    plt.plot(x, y_hat)
    plt.show()
