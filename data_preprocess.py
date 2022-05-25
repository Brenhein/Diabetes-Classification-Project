from sklearn.preprocessing import normalize
import numpy as np


def normal(x):
    """This function normalizes a data matrix"""
    return normalize(x, axis=0, norm='l2')


def convert_sample(line):
    """This function converts all the features for a sample into an int/float"""
    # Converts each feature for a sample into its respective type
    for i in range(len(line)):

        # Is the feature an integer or a float
        if i == 5 or i == 6:
            line[i] = float(line[i])
        else:
            line[i] = int(line[i])


def fill_missing(x_all):
    """This function fills in the 0s of the dataset if not certain indexes"""
    for c in range(1, len(x_all[0])):

        # Compute the averages of the column
        col = x_all[:, c]
        summation, length = 0, 0
        for v in col:
            if v != 0:
                summation, length = summation + v, length + 1
        avg = round(summation / length)

        # Fill the zeros with averages
        for i in range(len(col)):
            if not col[i]:
                col[i] = avg


def get_data(fname="diabetes_data.csv"):
    """This function parses the file and pulls out testing and traing samples"""
    fp, k = open(fname, "r"), 6
    
    # Gets the features and data
    x_all, y_all = [], []
    features = fp.readline().split(",")
    for line in fp:
        line = line[:-1].split(",")
        convert_sample(line)
        x_all.append(np.array(line[:-1]))
        y_all.append(line[-1])

    # Get rid of the zeros except zero pregnancies and normalize the dataset
    x_all = np.array(x_all)
    fill_missing(x_all)
    x_all = normal(x_all)

    # Partition data into k-folds
    fold_len = len(x_all) // k
    data, y = [], []
    for i in range(0, len(x_all), fold_len):
        data.append(x_all[i: i + fold_len])
        y.append(y_all[i: i + fold_len])

    # Goes through every subset
    x_train, x_test, y_train, y_test = [], [], [], []
    for i in range(len(data)):

        # This gets the current training and testing folds
        x_test_i, y_test_i = data.pop(i), y.pop(i)
        x_train_i, y_train_i = np.array([j for i in data for j in i]), np.array([j for i in y for j in i])
        data.insert(i, np.array(x_test_i))
        y.insert(i, np.array(y_test_i))

        # Add the data to the lists to return
        x_train.append(x_train_i)
        x_test.append(x_test_i)
        y_train.append(y_train_i)
        y_test.append(y_test_i)

    return x_train, x_test, y_train, y_test, features
