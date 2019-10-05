"""Udacity ML.10 Softmax()
"""
import numpy as np
import matplotlib.pyplot as plt

SCORES = np.array([3.0, 1.0, 0.2])

def softmax(x):
    """ Compute softmax values for each sets of SCORES in x.
    """
    # pass  # TODO: Compute and return softmax(x)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# print("softmax(score) : \n",softmax(SCORES/10))
# print("softmax(score) : \n",softmax(SCORES*10))

def display_graph_softmax():
    """ show the probablity layed between 0.0 ~ 1.0
    #  The graph shows how probablity changed
    """
    # Plot softmax curves
    x = np.arange(start=-2.0, stop=6.0, step=0.1)        # range for 'x'
    SCORES = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

    print(SCORES)
    # np.ndarray = [        ..... 3개 후보의 확률이 역전 된다.
    # [-2.  -1.   0.   1.   2.   3.   4.   5. ]     ..... 'x'
    # [ 1.   1.   1.   1.   1.   1.   1.   1. ]     ..... ones_like(x)
    # [ 0.2  0.2  0.2  0.2  0.2  0.2  0.2  0.2]]    ..... 0.2 * ones_like(x)

    plt.plot(x, softmax(SCORES).T, linewidth=2)
    plt.title("SoftMax function (x = -2 ~ 6)")
    plt.legend()
    plt.grid()
    plt.show()
display_graph_softmax()
