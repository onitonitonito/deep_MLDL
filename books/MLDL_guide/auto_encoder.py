"""
* 대표적 비지도 학습법 - 오토인코더, 3 min - p.138
"""
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

DIRS = os.path.dirname(__file__).partition("deep_MLDL\\")
ROOT = DIRS[0] + DIRS[1]


minst = input_data.read_data_sets(ROOT + "/_statics/mnist_data/", one_hot=True)

LEARN_RATE = 0.01
TRAIN_EPOCH = 20
BATCH_SIZE = 100
N_HIDDEN = 256
N_INPUT = 28 * 28

X = tf.placeholder(tf.float32, (None, N_INPUT))


""" 인코더 파트
*
"""
W_encode = tf.Variable(tf.random_normal(N_INPUT, N_HIDDEN))
b_encode = tf.Variable(tf.random_normal(N_HIDDEN))
encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))


""" 디코터 파트
*
"""
W_decode = tf.Variable(tf.random_normal(N_HIDDEN, N_INPUT))
b_decode = tf.Variable(tf.random_normal(N_INPUT))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode), b_encode))


cost = tf.reduce_mean(tf.pow(X - decoder, 2))
optimizer = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(int)

total_batch = int(mnist.train.num_examples / BATCH_SIZE)

for epoch in range(TRAIN_EPOCH):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs})

        total_cost += cost_val

    print('Epoch', '%04d' % (epoch + 1),
        'Avg.Cost =', '{:4f}'.format(total_cost/total_batch))

print('.... 학습완료! ....')


sample_size = 10
samples = sess.run(decoder, feed_dict={X: mnist.test.images[:sample_size]})

fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()

    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
    ax[1][i].imshow(np.reshape(samples[i], (28,28)))

plt.show()
