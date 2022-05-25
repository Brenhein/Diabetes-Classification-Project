import data_preprocess as dpp
from KNN import knn
from sklearn.metrics import confusion_matrix
from logistic_regression import logistic_regression
from nonlinear_svm import nonlinear_svm
from parzen import parzen
import plotting


PROMPT = \
'''Choose one of the following options for predicting diabetes

[1] K-NN
[2] Logistic Regression
[3] Gaussian Parzen Window
[4] Nonlinear SVM
[5] Quit

Enter an option: '''


def error(hp_name, hp_val, cms):
    """Calculates the confusion matrix and error for predicted and actual values"""
    cm_all = sum(cms)
    e = sum([cm_all[i][j] for i in range(len(cm_all)) for j in range(len(cm_all[i])) if i != j]) \
        / sum([cm_all[i][j] for i in range(len(cm_all)) for j in range(len(cm_all[i]))])
    print("\n{}={} error={}%\n".format(hp_name, hp_val, round(e, 3)))
    print(cm_all, end="\n\n")
    return e


def knn_main(train_x, train_y, test_x, test_y):
    """Wrapper function for K-NN before actually calling the model's file. This is so
    everything can properly be set up using cross validation before calling our model"""

    start = int(input("Enter a starting k: "))
    end = int(input("Enter an ending k: ")) + 1
    step = int(input("Enter a step size: "))
    
    # Test K-NN for range of k
    k_x, err_y = [], []
    for k in range(start, end, step):
        k_x_fold, predicted_y_fold, err_y_fold = [], [], []
        cms = []
        for inst in range(len(train_x)):

            # Gets the specific training and testing sets for a fold
            train_xi = train_x[inst]
            test_xi = test_x[inst]
            train_yi = train_y[inst]
            test_yi = test_y[inst]

            # Use K-NN and handle return
            predicted = knn(k, train_xi, train_yi, test_xi, test_yi)
            predicted_y_fold.append(predicted)
            cm = confusion_matrix(predicted, test_yi)
            cms.append(cm)

        # Append our average error for ALL folds
        err_y.append(error("K", k, cms))
        k_x.append(k)
    
    # Plot the error
    plotting.plot_error(k_x, err_y, "K", "% Error", "Error for K-NN", "K")


def logistic_regression_main(train_x, train_y, test_x, test_y):
    """Wrapper function for logistic regression before actually calling the model's file. This is so
    everything can properly be set up using cross validation before calling our model"""
    cms = []
    n = float(input("Enter a step size: "))
    for inst in range(len(train_x)):

        # Gets the specific training and testing sets for a fold
        train_xi = train_x[inst]
        test_xi = test_x[inst]
        train_yi = train_y[inst]
        test_yi = test_y[inst]

        # Use K-NN and handle return
        predicted = logistic_regression(n, train_xi, train_yi, test_xi, test_yi)
        cm = confusion_matrix(predicted, test_yi)
        cms.append(cm)

    # Append our average error for ALL folds
    error("n", n, cms)
    

def nonlinear_svm_main(train_x, train_y, test_x, test_y):
    """This function handles the set up and reporting of nonlinear SVM
    using a Gaussian Radial Basis Function kernel"""
    sigmas = input("Enter a list of sigmas, seperated by a comma: ")
    sigmas = [float(s.strip()) for s in sigmas.split(",")]

    # Test K-NN for range of k
    svm_x, err_y = [], []
    for s in sigmas:
        svm_x_fold, predicted_y_fold, err_y_fold = [], [], []
        cms = []
        for inst in range(len(train_x)):
            # Gets the specific training and testing sets for a fold
            train_xi = train_x[inst]
            test_xi = test_x[inst]
            train_yi = train_y[inst]
            test_yi = test_y[inst]

            # Use K-NN and handle return
            predicted = nonlinear_svm(s, train_xi, train_yi, test_xi)
            predicted_y_fold.append(predicted)
            cm = confusion_matrix(predicted, test_yi)
            cms.append(cm)

        # Append our average error for ALL folds
        err_y.append(error("σ", s, cms))

    # Plot the error
    plotting.plot_error(sigmas, err_y, "σ", "% Error", "Nonlinear RBF SVM Error", "σ")


def parzen_main(train_x, train_y, test_x, test_y):
    """Wrapper function for Gaussian Parzen Windows before actually calling the model's file, so
    everything can properly be set up using cross validation before calling our model"""

    sizes = input("Enter a list of window sizes, seperated by a comma: ")
    sizes = [float(s.strip()) for s in sizes.split(",")]

    # Test K-NN for range of k
    err_y = []
    for s in sizes:
        pw_x_fold, predicted_y_fold, err_y_fold = [], [], []
        cms = []
        for inst in range(len(train_x)):
            # Gets the specific training and testing sets for a fold
            train_xi = train_x[inst]
            test_xi = test_x[inst]
            train_yi = train_y[inst]
            test_yi = test_y[inst]

            # Construct a new matrix of the training samples, organized by class
            train_xp = [[], []]
            for i in range(len(train_xi)):
                train_xp[0].append(train_xi[i]) if train_yi[i] == 0 else train_xp[1].append(train_xi[i])

            # Construct a new matrix of the training samples, organized by class
            test_xp = [[], []]
            for i in range(len(test_xi)):
                test_xp[0].append(test_xi[i]) if test_yi[i] == 0 else test_xp[1].append(test_xi[i])

            # Use the Parzen Window technique
            actual, predicted = parzen(train_xp, test_xp, s, 8)
            predicted_y_fold.append(predicted)
            cm = confusion_matrix(predicted, actual)
            cms.append(cm)

        # Append our average error for ALL folds
        err_y.append(error("Sigma", s, cms))

    # Plot the error
    plotting.plot_error(sizes, err_y, "h", "% Error", "Gaussian Parzen Window Error", "h")


def main():
    train_x, test_x, train_y, test_y, features = dpp.get_data()
    
    # Command loop to test different models
    com = int(input(PROMPT))
    while com != 5:
        
        # Handle K-NN
        if com == 1:
            knn_main(train_x, train_y, test_x, test_y)
        
        # Handles Logistic Regression
        elif com == 2:
            logistic_regression_main(train_x, train_y, test_x, test_y)

        # Handles Gaussian Parzen Windows
        elif com == 3:
            parzen_main(train_x, train_y, test_x, test_y)

        # Handles Nonlinear SVM
        elif com == 4:
            nonlinear_svm_main(train_x, train_y, test_x, test_y)

        # Handles errors
        else:
            print("Invalid Command")
        
        com = int(input(PROMPT))


if __name__ == "__main__":
    main()
