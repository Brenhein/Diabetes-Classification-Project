from sklearn import svm
import numpy as np


def create_rbf_matrix(sig, x, y):
    """This function creates the kernel matrix with a gaussian radial basis function"""
    dim = len(y)
    x_rbf = np.zeros((dim, dim))
    for r1 in range(dim):
        for r2 in range(dim):

            # Calculate the RBF for train[r][c] and add to matrix
            diff = np.subtract(x[r1], x[r2])
            d = np.linalg.norm(diff) ** 2
            x_rbf[r1][r2] = np.exp(((-1 * d) / (2 * sig ** 2)))

    return x_rbf


def solve_nonlinear_svm(train_x, train_y, test_x, sig):
    """Solve nonlinear svm using quadratic programming with the kernel matrix
    The development of this function was aided by:
    https://scikit-learn.org/stable/modules/svm.html"""
    model = svm.SVC(kernel='rbf', gamma=sig)
    model.fit(train_x, train_y)

    # Predict values of the test set
    predictions = model.predict(test_x)
    return predictions


def nonlinear_svm(sig, train_x, train_y, test_x):
    """This function is the landing function for SVM, where the model will be
    trained and then tested"""

    # Multiply the kernel matrix by the output
    predicted = solve_nonlinear_svm(train_x, train_y, test_x, sig)
    return predicted
