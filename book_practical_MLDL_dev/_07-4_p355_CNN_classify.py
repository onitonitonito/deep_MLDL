""" p.356 - 
"""
import os
import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense

# '루트'와 '작업'디렉토리 설정 - for 스크립트런
DIRS = os.path.dirname(__file__).partition("deep_MLDL")
ROOT = DIRS[0] + DIRS[1]

#분류 대상 카테고리 선택하기  ...  (*1)
caltech_dir = os.path.join(ROOT, "_static", "image", "category_101objects", "")
load_dir = os.path.join(ROOT, "_static", "image", "")
categories = ['chair', 'camera', 'butterfly', 'elephant', 'flamingo']

#하이퍼 파라메터
nb_classes = len(categories)        # 5개의 카테고리

# 이미지 크기 지정  ...  (*2) : [64x64]
image_w = 64
image_h = 64

# 데이터 열기 .. deep_MLDL\_static\image\object_5.npy
X_train, X_test, y_train, y_test = np.load(load_dir + "object_5.npy")

# 데이터 정규화 하기 .. Generalize data
X_train = X_train.astype("float") / 256
X_test = X_test.astype("float") / 256
print("X_train.shape :", X_train.shape)         # [250, 64. 64]

# 모델 구축하기  ...  (*2)  : p.356
model = Sequential()
model.add(Convolution2D(32, 3, 3,
    border_mode='same',
    input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())        # ... (*3)  : p.356
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.complie(loss = 'binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

# 모델 트레이닝하기  ...  (*4)  :.p356
model.fit(X_train, y_train, batch_size=32, nb_epoch=50)

# 모델 평가하기  ...  (*5)
score = model.evaluate(X_train, y_train)
print("lose :", score[0])
print("accuracy :", score[1])

# 일단은 에러, 옵션으로 다시 불러보기..  ing ...
