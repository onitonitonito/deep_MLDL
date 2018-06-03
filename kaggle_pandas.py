import numpy as np          # linear algebra
import pandas as pd         # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import sys

# from os.path import dirname, exists
ROOT = 'four_pillars_of_oop\\'
DIRS = os.path.dirname(__file__).partition(ROOT)
ROOT_DIR = DIRS[0] + DIRS[1]

print(ROOT_DIR+"_static")
for i, filename in enumerate(os.listdir(ROOT_DIR+"_static"), 1):
    print("%s. %s"%(i, filename))
