import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


# '루트'와 '작업'디렉토리 설정 - for 스크립트런
import os
DIRS = os.path.dirname(__file__).partition("k_mooc_reboot\\")
ROOT = DIRS[0] + DIRS[1]

#분류 대상 카테고리 선택하기  ...  (*1)
file_dir = os.path.join(ROOT, "_static", "image", "category_101objects","")
categories = ['chair', 'camera', 'butterfly', 'elephant', 'flamingo']


# 글롭은 해당 DIR의 fname_with_dir을 '리스트'로 반환한다
files = glob.glob(file_dir + "*.jpg")

for f in files:
    print (f)
