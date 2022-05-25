import numpy as np
import math


def predict(x, w):
    """
    This function uses a sample and its weights to predict its values
    """
    x, w = np.array(x), np.array(w)
    return float(np.dot(x, w))


def sigmoid(x, w):
    """
    Calculates the output of the sigmoid function
    """
    p = round(predict(x, w), 3)
    f = 1 / (1 + math.exp(-p)) if p >= 0 else math.exp(p) / (1 + math.exp(p))
    return f


def error_gradient(w, train_x, train_y):
    """
    This function calculates the error of the gradient with respect to w
    """
    error = 0
    for n in range(len(train_x)):
        xn, yn = train_x[n], train_y[n]
        y_hat = sigmoid(xn, w)
        error += (y_hat - yn) * np.array(xn)
    return error


def gradient_descent(n, train_x, train_y, maxiter=1000):
    """
    This function performs gradient descent to find the optimal vector of 
    of weights to use for logistic regression
    """
    wt = len(train_x[0]) * [0]
    diff, i = 1, 0
    
    # Update the weigths for each step of the first order gradient
    while diff > .001 and i < maxiter:
        wt_old = wt
        i += 1
        
        # Move the gradient down
        e_grad = error_gradient(wt, train_x, train_y)
        wt = np.round_(wt - n * e_grad, 5)
        
        # Are we close enough to say we converged
        wt_diff = wt - wt_old
        diff = np.linalg.norm(wt_diff)

    return wt
        

def logistic_regression(n, train_x, train_y, test_x, test_y):
    wt = gradient_descent(n, train_x, train_y)

    # Now test the data using the estimated weights
    predicted = []
    for i in range(len(test_x)):
        x, y = np.array(test_x[i]), test_y[i]
        predicted.append(round(sigmoid(wt, x)))

    return predicted

