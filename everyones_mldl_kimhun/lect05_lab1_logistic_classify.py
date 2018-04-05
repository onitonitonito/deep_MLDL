import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def test1_simple_sigmoid(x_data, y_data, num_x, learning_rate=1e-2):
    X = tf.placeholder(tf.float32, shape=[None, num_x])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    """ X:[n,2] x W:[2,1] + b:[1] = Y:[n,1]
     - n = 6
     - nb_classes = 1
    """
    W = tf.Variable(tf.random_normal(shape=[num_x,1]))
    b = tf.Variable(tf.random_normal(shape=[1]))

    """ sigmoid using hypothesis :
      - H(x) = 1 / (1+e**(-W_t.X)) = tf.div(1., 1.+tf.exp(tf.matmul(X,W)+b))
      - cost(W) = -1/m * sum(y * log(H(x))) + (1-y) * (log(1 - H(x)))
    """
    hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost)


    """ Accuracy computation / True if hypothesis > 0.5 else False """
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32, name='predicted')
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


    """ Train Model """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(10001):
            cost_val, _ = sess.run([cost, train], feed_dict = {X:x_data, Y:y_data})
            if step%200 == 0:
                print("{:5}. __ cost:{:.6f}, {}".format(step, cost_val, _))

        hypo_val, Pred_val, accu_val = sess.run(
            [hypothesis, predicted, accuracy],
            feed_dict = {X:x_data, Y:y_data})

        print("\nHypothesis: \n{} \n\npredicted: \n{} \n\naccuracy: {}".format(
            hypo_val,
            Pred_val,
            accu_val))

""" simplesigmoid solving
https://www.youtube.com/watch?v=2FeWGgnyLSw&feature=youtu.be
# x 갯수 = 2개 의 질문 일때,
# print(x_data)     # None, (1,2)
# print(y_data)     # None, (1,1)
"""
x_data = np.array([[n
    for n in range(0+i, 2+i)]
        for i in range(0, 11, 2)])

y_data = np.array([[0]
    if n < 3 else [1]
        for n in range(6)])

print(x_data)     # None, (1,2)
print(y_data)     # None, (1,1)
"""
[[ 0  1]
 [ 2  3]
 [ 4  5]
 [ 6  7]
 [ 8  9]
 [10 11]]

[[0]
 [0]
 [0]
 [1]
 [1]
 [1]]
"""

# test1_simple_sigmoid(x_data, y_data, num_x=2, learning_rate=1e-2)


""" classify diabetes : 후반부 유튜브
https://www.youtube.com/watch?v=2FeWGgnyLSw&feature=youtu.be
x 갯수 = 8개의 질문일때,
"""
xy = np.loadtxt('./_csv_hunkim/data03_diabetes.csv',
    delimiter=',',
    dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# test1_simple_sigmoid(x_data, y_data, num_x=8, learning_rate=2e-1)
