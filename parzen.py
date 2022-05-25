"""
The initial file was created by me to be used for handwritten digit classification. However, alterations were
made to make it work with the current dataset. This is being tested, as it has worked well in the based for
certain classification tasks.
"""
from operator import itemgetter
import numpy as np
import scipy.stats as sp


def hn(train, h, x, d):
    """For a given set of points (from a normal distribution), return the resulting
    estimatiated denisity for a given x using a gaussian kernel function"""
    dist, densities = sp.multivariate_normal(np.zeros(d), np.identity(d)), []

    # estimate the density of x using generated points
    for w, patterns in enumerate(train):
        px = 0
        for v in patterns:
            px += dist.pdf((np.subtract(x, v))/h)
        densities.append((px / (len(train) * (h ** d)), w))

    return densities


def parzen(train, test, h, d):
    """Given a window size, this function uses a gaussian parzen window to
    estimate the density of a testing point, given a cluster of training points"""
    predicted, actual = [], []
    for w, patterns in enumerate(test):
        for i, pat in enumerate(patterns):
            d_est = hn(train, h, pat, d)
            best = max(d_est, key=itemgetter(0))[1]
            predicted.append(best)
            actual.append(w)
    return actual, predicted
