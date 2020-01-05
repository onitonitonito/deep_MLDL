"""
# 2.1 신경망과의 첫만남
"""
# mnist 데이터샛을 내장하고 있다.


from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.datasets import mnist

print(__doc__)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# for TES
print(type(train_images))       # <class 'numpy.ndarray'>
print((train_labels)[17])
print((test_images)[17][17])
print(test_labels.shape)

# 데이터 이미지셋 준비
train_images = train_images.reshape((60_000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10_000, 28*28))
test_images = test_images.astype('float32')/255

#레이블 준비하기
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 신경망 모델작성
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

# 신경망 모델 컴파일
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'],
                )

# 모델 피팅(훈련) 시작
network.fit(x=test_images, y=test_labels, epochs=1, batch_size=128)

"""
Using TensorFlow backend.

# 2.1 신경망과의 첫만남

2019-12-02 16:52:35.569976: I tensorflow/core/platform/cpu_feature_guard.cc:142]
Your CPU supports instructions that this TensorFlow binary was not compiled to
use: AVX2

Epoch 1/1
  128/10000 [..............................] - ETA: 6s - loss: 2.3838 - accuracy: 0.0781
 1024/10000 [==>...........................] - ETA: 1s - loss: 1.3631 - accuracy: 0.5879
 1792/10000 [====>.........................] - ETA: 0s - loss: 1.0646 - accuracy: 0.6830
 2560/10000 [======>.......................] - ETA: 0s - loss: 0.9243 - accuracy: 0.7305
 3328/10000 [========>.....................] - ETA: 0s - loss: 0.8119 - accuracy: 0.7662
 4096/10000 [===========>..................] - ETA: 0s - loss: 0.7650 - accuracy: 0.7776
 4864/10000 [=============>................] - ETA: 0s - loss: 0.7065 - accuracy: 0.7969
 5760/10000 [================>.............] - ETA: 0s - loss: 0.6584 - accuracy: 0.8109
 6528/10000 [==================>...........] - ETA: 0s - loss: 0.6205 - accuracy: 0.8220
 7296/10000 [====================>.........] - ETA: 0s - loss: 0.5933 - accuracy: 0.8298
 8064/10000 [=======================>......] - ETA: 0s - loss: 0.5784 - accuracy: 0.8343
 8960/10000 [=========================>....] - ETA: 0s - loss: 0.5499 - accuracy: 0.8436
 9856/10000 [============================>.] - ETA: 0s - loss: 0.5258 - accuracy: 0.8493
10000/10000 [==============================] - 1s 75us/step - loss: 0.5219 - accuracy: 0.8505

Process returned 0 (0x0)        execution time : 5.318 s
계속하려면 아무 키나 누르십시오 . . .
"""
