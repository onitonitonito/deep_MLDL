import _script_run_utf8
_script_run_utf8.main()

import numpy as np          # linear algebra
import pandas as pd         # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import sys

# from os.path import dirname, exists
ROOT = 'deep_MLDL'
DIRS = os.path.dirname(__file__).partition(ROOT)
ROOT_DIR = DIRS[0] + DIRS[1]

print(ROOT_DIR+"_static")
for i, filename in enumerate(os.listdir(os.path.join(ROOT_DIR, "_static")), 1):
    print("%s. %s"%(i, filename))
