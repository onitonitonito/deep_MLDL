"""
# os.getcwd()는 script_run 에서 다른결과를 낸다
# __file__ = 현재위치는 __file__에서 추출하는게 좋다.
"""
# ------ root path 를 sys.path.insert 시키는 코드 ... 최소 4줄 필요------
import os, sys                                                      # 1
top = "deep_MLDL"                                                   # 2
root = "".join(os.path.dirname(__file__).partition(top)[:2])+"\\"   # 3
sys.path.insert(0, root)                                            # 4
# ---------------------------------------------------------------------

import _assets.script_run
from _assets.configs_global import dir_assets, dir_statics

print(__doc__)

# 화일을 소팅한다.

for i, filename in enumerate(os.listdir(root + "everyones_mldl_kimhun")):
    print(f"{i+1:02}. {filename}")
