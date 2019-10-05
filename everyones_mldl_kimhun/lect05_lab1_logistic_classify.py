import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# '루트'와 '작업'디렉토리 설정 - for 스크립트런
DIRS = os.path.dirname(__file__).partition("deep_MLDL")
ROOT = DIRS[0] + DIRS[1]
CSV_DIR = os.path.join(ROOT, "_statics", "_csv_hunkim", "")

# 항상 동일한 값을 위해 랜덤시드값을 고정
# tf.set_random_seed(777)  # ... accuracy = 0.5
tf.set_random_seed(743)  #  ... accuracy = 0.8333

def sigmoid(x):         # 시그모이드를 직접 계산하려면 이렇게 한다
    return 1 / (1+np.exp(-x))

def simple_sigmoid(x_data, y_data, num_x, learning_rate=1e-2):
    X = tf.placeholder(tf.float32, shape=[None, num_x])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

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

        for step in range(30_001):
            cost_val, _ = sess.run([cost, train], feed_dict = {X:x_data, Y:y_data})
            if step%200 == 0:
                print("{:5}. __ cost:{:.6f}, {}".format(step, cost_val, _))

        hypo_val, pred_val, accu_val = sess.run(
            [hypothesis, predicted, accuracy],
            feed_dict = {X:x_data, Y:y_data})

        print("\nHypothesis: \n", hypo_val)
        print("\npredicted: \n", pred_val)
        print("\naccuracy: ",accu_val)

        fW = W.eval(sess)
        fb = b.eval(sess)
    return fW, fb

""" simplesigmoid solving
https://www.youtube.com/watch?v=2FeWGgnyLSw&feature=youtu.be
# x 갯수 = 2개 의 질문 일때,
# print(x_data)     # None, (6,2)
# print(y_data)     # None, (6,1)
"""
x_data = np.array([[n
    for n in range(0+i, 2+i)]
        for i in range(0, 11, 2)])

y_data = np.array([[0]
    if n < 3 else [1]
        for n in range(6)])

""" tf 의 계산값은 세션을 실행해서 수행해야 함 (플레이스 홀더/변수의 차이)
# cost_val = 에러율 sess.run([cost, train], feed_dict={})
#    ... cost, train은 placeholder(Y)의 함수이므로 feed_dict을 먹여 줌
#    ... 반면, W/b 는 Variable() 이므로 eval(sess) 로 환산할 수 있음
#
# fW, fb 로 최종 계산한 값 = hypo_val = sigmoid(WX + b) = 0~1값:확률
# [[1] if hypo_val > 0.5 else [0]] = pred_val = 예측값은 1, 0
"""

fW, fb = simple_sigmoid(x_data, y_data, num_x=2, learning_rate=1e-4)
print(x_data.shape)     # None, (1,2)
print(x_data)
print(y_data.shape)     # None, (1,1)
print(y_data)

print()
print("W :\n", fW)
print("b :\n", fb)

print('\n\nRES=', sigmoid(np.add(np.dot(x_data, fW), fb)))
print('\n\nRES=', np.array([[1 if float(f)>0.5 else 0] for f in sigmoid(np.add(np.dot(x_data, fW), fb))]))

""" classify diabetes 당뇨판단(Yes/No): 후반부 유튜브
https://www.youtube.com/watch?v=2FeWGgnyLSw&feature=youtu.be
x 갯수 = 8개의 질문일때,
"""
# xy = np.loadtxt(CSV_DIR+'data03_diabetes.csv', delimiter=',', dtype=np.float32)
# x_data = xy[:, 0:-1]
# y_data = xy[:, [-1]]
#
# fW, fb = simple_sigmoid(x_data, y_data, num_x=8, learning_rate=2e-1)
# print(x_data.shape)     # None, (1,2)
# print(y_data.shape)     # None, (1,1)
#
# print()
# print("W :\n", fW)
# print("b :\n", fb)
#
# print('\n\nRES=', sigmoid(np.add(np.dot(x_data, fW), fb)))
# print('\n\nRES=', np.array([[1 if float(f)>0.5 else 0] for f in sigmoid(np.add(np.dot(x_data, fW), fb))]))
