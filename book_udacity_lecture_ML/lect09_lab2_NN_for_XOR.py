import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

""" XOR Neural Network - Problem
#
# x:[4,2]* w:[2,1] + b:[1] = y:[4,1] / nb_class (answer) = 1
"""
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)

W = tf.Variable(tf.random_normal(shape=[2,1], name='weight'))
b = tf.Variable(tf.random_normal(shape=[1], name='bias'))

hypothesis = tf.sigmoid(x=tf.matmul(X,W)+b, name='sigmoid_func')
cost = tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y)* tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

""" Run graph """
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter('./_logdir', sess.graph)

    costs = []
    for step in range(10001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})

        if not step%100:
            print("{:5}__ Cost:{} / W:".format(
                step,
                sess.run(cost, feed_dict={X:x_data, Y:y_data}),
                sess.run(W)))
            costs.append(sess.run(cost, feed_dict={X:x_data, Y:y_data}))

    hypo_val, cost_val, accu_val = sess.run(
        [hypothesis, predicted, accuracy],
        feed_dict={X:x_data, Y:y_data})

    print("hypo_val={}, cost_val={}, accu_val={}".format(
        hypo_val, cost_val, accu_val))

    print(costs)

    plt.grid()
    plt.xlim(0, 100)
    plt.plot(costs)
    plt.show()
