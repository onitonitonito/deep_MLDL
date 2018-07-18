"""
# Fashion mnist -- sub
# https://github.com/zalandoresearch/fashion-mnist
#----------------------------------------------
# Fashion-MNIST is a dataset of Zalando's article images—consisting of a
# training set of 60,000 examples and a test set of 10,000 examples. Each example
# is a 28x28 grayscale image, associated with a label from 10 classes. We intend
# Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST
# dataset for benchmarking machine learning algorithms. It shares the same
# image size and structure of training and testing splits.
#
#
#\n\n\n"""
print(__doc__)


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
