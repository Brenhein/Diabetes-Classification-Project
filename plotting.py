import matplotlib.pyplot as plt
from operator import itemgetter


def plot_error(x, y, xax, yax, title, hp):
    i = min(enumerate(y), key=itemgetter(1))[0]
    min_k = x[i]
    
    plt.ylabel(yax)
    plt.xlabel(xax)
    plt.title(title)
    plt.plot(x, y)
    plt.axvline(x=min_k, color='g', linestyle='--', 
                label="Minimum % Error ({}={})".format(hp, min_k))
    plt.legend(loc="upper left")
    plt.savefig(xax + "_err.png")
    plt.show()
