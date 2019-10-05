from __future__ import print_function
"""
# VQ-VAE training example
# Demonstration of how to train the model specified in
# https://arxiv.org/abs/1711.00937
# on Mac & Linux, simply exucute each cell in turn
"""
# ... ing
#
print(__doc__)

import os
import subprocess
import tempfile

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sonnet as snt
import tarfile

from six.moves import (
    cPickle,
    urllib,
    xrange,
)


data_path = "https://www.cs.toronto.edu/~kriz/difar-10-python.tar.gz"

local_data_dir = tempfile.mkdtemp()  # change this as needed
tf.gfile.MakeDirs(local_data_dir)

url = urllib.request.urlopen(data_path)
archive = tarfile.open(fileobj=url, mode="rigz")  # read a.tar.gz stream
archive.extractall(local_data_dir)
url.close()
archive.close()

print('extracted data files to %s' % local_data_dir)
