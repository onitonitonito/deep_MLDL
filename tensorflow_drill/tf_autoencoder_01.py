"""
"""
import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# '루트'와 '작업'디렉토리 설정 - for 스크립트런
DIRS = os.path.dirname(__file__).partition("deep_MLDL")
ROOT = DIRS[0] + DIRS[1]
CSV_DIR = os.path.join(ROOT, "_statics","MNIST_data", "")

tf.set_random_seed(777)         # 동일한 재현성을 위하여, 시드 동일하게 초기화

mnist = input_data.read_data_sets(CSV_DIR, one_hot=True)
# mnist = input_data.read_data_sets("", one_hot=True)

# Hyper variable Setting
nb_classes = 10                 # 입력 변수의 갯수 = 10개
learning_rate = 1e-2            # 미소스텝 = 1 * 10**-2
training_epoches = 20           # 전체 데이터 반복횟수
batch_size = 100                # 읽어 들이는 배치값의 크기

n_hidden = 256
n_input = 28*28                 # 784 pix. for 1 letter

# placeholder
X = tf.placeholder(tf.float32, [None, n_input])

# encoder
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))   # We:[784, 256]
b_encode = tf.Variable(tf.random_normal([n_hidden]))            # be:[256]
encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))

# decoder
W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))   # Wd:[256, 784]
b_decode = tf.Variable(tf.random_normal([n_input]))             # bd:[784]
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode), b_decode))

cost = tf.reduce_mean(tf.pow(X-decoder, 2))
train = tf.train.RMSPropOptimizer(learning_rate)                # RMS옵티마이져
optimizer = train.minimize(cost)

# run graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

num_examples = mnist.train.num_examples
# MNIST 에 저장된, 총 학습 샘플의 갯수 = 550k
# 한 번에 읽어들이는 배치값 = 100개 (1배치 = 100글자)
# 한번의 Epoch 에 필요한 총 배치횟수 = 500k / 100 = 550번 배치를 읽어들임.
# 학습을 위해서 20번 반복할 예정 = 트레이닝 에폭 = 20회
total_batch = int(num_examples / batch_size)        # 550k / 100 = 550

f = open("./write_result/tf_autoencode_echo.pdb", "w")

for epoch in range(training_epoches):               # 20번 반복학습
    total_cost = 0

    # 1에폭을 학습하기 위해, 550배치(550k글자) - 1배치에 100글자씩 학습한다.
    for i in range(total_batch):            # 550 배치 = 1 에폭
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X:batch_xs})
        total_cost += cost_val              # 코스트값 누적 / 총배치 횟수(550번) = 평균cost

    avg_cost = total_cost/total_batch
    f.write("%10.6f"%(avg_cost)+",\n")
    print("Epoch: %2s __ Avg.Cost: %10.7f"% (epoch + 1, avg_cost))

print("... Learning Finished! ...   TOTAL COST: %s"% total_cost)
f.close()

# test set 10 samples
sample_size = 10
samples = sess.run(
    decoder,
    feed_dict={X:mnist.test.images[:sample_size]})

fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))
for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()

    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
    ax[1][i].imshow(np.reshape(samples[i], (28,28)))
plt.show()
sess.close()
