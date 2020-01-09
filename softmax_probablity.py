"""
# Softmax Probablity calculation!
"""
# print(__doc__)

import numpy as np

scores = [
    7.0,
    3.0,
    1.0,
    0.1,
    ]
probs_int, probs_float = [], []

def main():
    tuples_int_float = softmax(scores)
    for i, _ in enumerate(scores):
        print(f"Score {scores[i]} -> ", end='')
        print(f"{tuples_int_float[1][i]} -> ", end='')
        print(f"{tuples_int_float[0][i]}")
    pass

def softmax(scores):
    """  """
    for score in scores:
        prob = np.exp(score) / np.exp(scores).sum()
        probs_float.append(prob)
        probs_int.append(round(prob, 2))
    return probs_int, probs_float





if __name__ == '__main__':
    main()
