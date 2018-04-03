""" 트레이닝 / 테스트 SET & 데이터 노말라이즈
# X:[n*3] * W:[3*1] + b[1] = Y:[n, 1] - 선형회귀:스코어 계산
# ... 작성중, 수정 필요 ...
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_data = np.array(
    [
        [1.,2.,3.],
        [4.,3.,3.],
        [7.,5.,4.]])
y_data = np.array([[12.],[32.],[10.]])

""" SET VARIABLES : X, Y, W ,b
# H(x) = X*W + b = Y (hypothesis)
"""
X = tf.placeholder(dtype="float32", shape=[None, 3], name="X")
Y = tf.placeholder(dtype="float32", shape=[None, 1], name="Y")

W = tf.Variable(initial_value=tf.random_normal([3,1]), name="weight")
b = tf.Variable(initial_value=tf.random_normal([1]), name="bias")

""" Linear Regression (Logostics Regression)
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
"""

""" Multi-Nomial Classification : Softmax Regression
# Using : cost() = Cross-Entropy Function
"""
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

""" correct prediction test model """
prediction = tf.argmax(hypothesis, 1)               # 1-hot
is_correct = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf. reduce_mean(tf.cast(is_correct, tf.float32))

""" RUN THE GRAPH : SESSION """
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, train_val, hypo_val = sess.run(
            [cost, train, hypothesis],
            feed_dict={X:x_data, Y:y_data})

        if step%100 == 0:
            print("cost:%8.3f ____ train: %s, %7.4f " %(
                cost_val,
                train_val,
                float(hypo_val[0])))
