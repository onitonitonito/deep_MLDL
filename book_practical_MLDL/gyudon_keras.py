import os
import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils

DIRS = os.path.dirname(__file__).partition("k_mooc_reboot\\")
ROOT = DIRs[0] + DIRS[1]

# 분류대상 카테고리
FILEDIR = ROOT + "_static/image/gyudon/"
CATEGORIES = ["normal", "beni", "negi", "cheese",]
nb_classes = len(CATEGORIES)
image_size = 50

# np로 변환한 학습 데이터 다운로드 하기 --- (*1)
def main():
    X_train, X_test, y_train, y_test = np.load("./gyudon.npy")

    #데이터 정규화 하기
    X_train = X_train.astype("float") / 256
