""" '이미지'를 '숫자' 데이터로 변환 시키는 프로그램
이미지 데이터를 '넘파이' 형식으로 변환
"""
import os
import glob
import numpy as np

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from PIL import Image

# 분류 대상 카테코리 --- (*1)
ROOT_DIR = "./image/"
categories = ["normal", "beni", "negi", "cheese"]
nb_classes = len(categories)
image_size = 50

# 폴더마다의 이미지 데이터 읽어들이기 --- (*2)
X = []      # 이미지 데이터
Y = []      # 레이블 데이터

for idx, cat in enumerate(categories):
    lmage_dir =  ROOT_DIR + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    print("----", cat, "처리중")

    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")                    # 색상모드 변경
        img = img.resize((image_size, image_size))  # 이미지 크기 변경
        data = np.asarray(img)
        X.append(data)
        Y.append(idx)
X = np.array(X)
Y = np.array(Y)

# 학습 전용데이터와 테스트 전용 데이터를 분리한다 -- (*3)
X_train, X_test, y_train, y_test = train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./image/gyudon.npy")
print("O.K.,", len(Y))

"""
(!)주의 = 사진화일을 분류한 다음 실행할 것.
"""
