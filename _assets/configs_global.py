"""
# _assets/configs_global.py - 각종변수 및 Path를 지정
"""
import script_run
import os
import sys
print(__doc__)


name_top = "deep_MLDL"

# 폴더 구조관련 변수
dir_top = "deep_MLDL"
dir_array = os.path.dirname(__file__).partition(dir_top)
dir_root = "".join(dir_array[:2]) + "\\"

dir_assets = dir_root + "_assets\\"
dir_statics = dir_root + "_statics\\"



dir_dict = {
    '_assets' : ['_assets',],
    '_statics' : ['_statics',],
}

def main():
    dir_top = get_dir_by_name(name_top)
    print("top :", dir_top)
    print("_assets :", get_dir(dir_top, dir_dict, "_assets"))
    print("_statics :", get_dir(dir_top, dir_dict, "_statics"))


def get_dir_by_name(name_top):
    """
    # top level name을 기준으로 dir을 반환한다.
    # 사용환경에 따라, 실행기준이 달라지기 때문에
    # 탑레벨을 시스템 path 에 추가한다.
    """
    dir_array = os.path.dirname(__file__).partition(name_top)
    dir_by_name = "".join(dir_array[:2])
    return dir_by_name

def join_dir(*dirs_array):
    return os.path.join(*dirs_array)

def get_dir(dir_top, dir_dict, key_name):
    return os.path.join(dir_top, *dir_dict[key_name])


if __name__ == '__main__':
    main()
