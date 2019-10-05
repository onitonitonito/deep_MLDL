import script_run

import numpy as np          # linear algebra
import pandas as pd         # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import sys

# from os.path import dirname, exists
ROOT = 'deep_MLDL'
DIRS = os.path.dirname(__file__).partition(ROOT)
ROOT_DIR = DIRS[0] + DIRS[1]

print(ROOT_DIR+"_statics")
for i, filename in enumerate(os.listdir(os.path.join(ROOT_DIR, "_statics")), 1):
    print("%s. %s"%(i, filename))
