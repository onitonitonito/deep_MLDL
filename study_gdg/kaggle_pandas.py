"""
# os.getcwd()는 script_run 에서 다른결과를 낸다
# __file__ = 현재위치는 __file__에서 추출하는게 좋다.
"""
import os, sys
sys.path.insert(0, os.path.abspath(".."))

[print(path) for path in sys.path]
quit()

import numpy as np          # linear algebra
import pandas as pd         # data processing, CSV file I/O (e.g. pd.read_csv)

import _assets.configs_global

print(__doc__)

# print(DIRS)               # for TEST.
print(ROOT_DIR + "_statics")

# 화일을 소팅한다.

for i, filename in enumerate(os.listdir(ROOT_DIR + "_assets")):
    print(f"{i+1:02}. {filename}")


print(na)
