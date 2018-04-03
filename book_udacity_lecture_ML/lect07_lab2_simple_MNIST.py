""" Simple MNIST Dataset test
# Check out https://www.tensorflow.org/get_started/mnist/beginners
# for more information about the mnist dataset
# MNIST data : 1-image of shape = 28 * 28 = 784 pixel * 10 = 7,840 pixel
# Training = 15-Epoches
# 1 batch = 100 px. / 1-Epoch = 78.4 batches / 15-Epoch = 1,176 batches
"""
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

""" Set parameter : """
learning_rate = 1e-1
nb_classes = 10
training_epoches = 15
batch_size = 100
display_epoch = 1
logdir = './_logdir/'

""" MNIST image of shape = 28 * 28 = 784 pixel
# X:[n, 784] * W:[784, 10] + b[10] = Y:[n, 10]
# X:n = 트레이닝 데이타의 겟수 / Y:n = 질문의 갯수가 될 수 있다.
# 784개의 데이터를 읽어야, 0~9 중 택1을 할수 있다 : softmax().argmax()
"""
X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="X" )
Y = tf.placeholder(dtype=tf.float32, shape=[None, nb_classes], name="Y" )

W = tf.Variable(tf.random_normal(shape=[784, nb_classes], name="weight"))
b = tf.Variable(tf.random_normal(shape=[10], name="bias"))

""" Softmax = Sigmoid(y) = softmax(H(x)) ... H(x) = X.W + b
# cost() = 크로스엔트로피 = Sum(i=1~j)(Y * log())
"""
hypothesis = tf.nn.softmax(tf.matmul(X,W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
train =  optimizer.minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./_logdir', sess.graph)


    for epoch in range(training_epoches):
        average_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # total_batch = 550, num_examples = 55k

        # train(cost minimize) = 100 x 550 repeats = 1 Epoch = whole data set
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cost_val = sess.run(
                fetches=[train, cost],
                feed_dict={X:batch_xs, Y:batch_ys})
            average_cost += cost_val / total_batch

        if (epoch+1) % display_epoch == 0:
            print("Epoch:{:3d} ___ Cost:{:.9f}".format(
                epoch + 1,
                average_cost))
    print("Learning finished")          # whole 데이터 셋을  15번 반복학습

    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(
        session=sess,
        feed_dict={X: mnist.test.images, Y: mnist.test.labels}))


    while True:
        # Get one and predict
        r = random.randint(0, mnist.test.num_examples - 1)
        print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
        print("Prediction: ", sess.run(
            tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

        plt.imshow(
            mnist.test.images[r:r + 1].reshape(28, 28),
            cmap='Greys',
            interpolation='nearest')
        plt.show()

        if input("\n\n\nSTOP? (Y/N)").lower().startswith('y'):
            break
