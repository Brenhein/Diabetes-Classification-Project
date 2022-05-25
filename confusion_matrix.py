from sklearn.metrics import confusion_matrix


def confusion(predicted, test_y):
    """This function calculates a confusion matrix for our model"""
    cm = confusion_matrix(predicted, test_y)
    w = sum([cm[i][j] for i in range(len(cm)) for j in range(len(cm[i])) \
             if i != j]) / len(test_y)
    print("\nerror={}%\n".format(100 * round(w, 3)))
    print(cm, end="\n\n")
    return w
