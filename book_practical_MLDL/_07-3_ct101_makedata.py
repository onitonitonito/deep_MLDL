""" Caltech 101objects 를 파이썬 데이터로 변경하여 저장하기
 - caltech site : http://www.vision.caltech.edu/Image_Datasets/Caltech101/
 """
 # Description :
 # Pictures of objects belonging to 101 categories. About 40 to 800 images per
 # category. Most categories have about 50 images. Collected in September 2003
 # Fei-Fei Li, Marco Andreetto, and Marc 'Aurelio Ranzato.  The size of each
 # image is roughly 300 x 200 pixels.
 #
 # We have carefully clicked outlines of each object in these pictures, these are
 # included under the 'Annotations.tar'. There is also a matlab script to view
 # the annotaitons, 'show_annotations.m'.

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib.image import imread
from sklearn.model_selection import train_test_split

# '루트'와 '작업'디렉토리 설정 - for 스크립트런
DIRS = os.path.dirname(__file__).partition("k_mooc_reboot\\")
ROOT = DIRS[0] + DIRS[1]
WORK_DIR = os.path.join(ROOT, "_static", "")

# 스크립트런 '한글' 표시를 위한 커스텀 모듈 실행
sys.path.append(ROOT)
import _script_run_utf8
_script_run_utf8.main()

#분류 대상 카테고리 선택하기  ...  (*1)
caltech_dir = WORK_DIR + "image\\category_101objects\\"
categories = ['chair', 'camera', 'butterfly', 'elephant', 'flamingo']

# 정답은 5개 (nb_classes)
nb_classes = len(categories)

# 이미지 크기 지정  ...  (*2) : [64x64]
image_w = 64
image_h = 64
pixels = image_w * image_h * 3

# 이미지 데이터 읽어들이기  ...  (*3)
X = []
Y = []

for i, category in enumerate(categories):
    # 레이블 지정  ...  (*4)
    label = [0 for i in range(nb_classes)]          # 빈 답안지 = [0,0,0,0,0]
    label[i] = 1                                    # 레이블, 빈 답안지 위에 마킹

    # 이미지  ...  (*5) : 글롭은 해당dir의 fname_with_dir을 '리스트'로 반환 함.
    image_dir = caltech_dir + "/" + category
    files = glob.glob(image_dir + "/*.jpg")

    for idx, f in enumerate(files):
        img = Image.open(f)                     # ...  (*6) PIL.image object
        img = img.convert("L")                # ... 'L'=gray / 'RGB'=color
        img = img.resize((image_w, image_h))    # ... [223x300] --> [64x64]
        data = np.asarray(img)
        X.append(data)
        Y.append(label)

        if idx%10 == 0:
            print(idx, "\n", data)

X = np.array(X)
Y = np.array(Y)

# 학습 전용데이터 와 테스트 전용데이터를 구분한다  ...  (*7)
X_train, X_test, y_train, y_test = train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)

np.save(WORK_DIR + "image\\object_5.npy", xy)

print("O.K. ___ total data :  %s 개 완료!"% len(Y))
