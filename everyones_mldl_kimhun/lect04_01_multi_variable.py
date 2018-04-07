import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 디렉토리 설정 (ROOT, CSV_DIR, FNAME)
DIRS = os.path.dirname(__file__).partition("deep_MLDL")
ROOT = DIRS[0] + DIRS[1]
CSV_DIR = os.path.join(ROOT, "_static", "_csv_hunkim", "")
FNAME = "data01_test_score.csv"

""" Hyper Parameters """
REPEATATION = 17_001
LEARNING_RATE = 1e-5


""" Making Data set """
tf.set_random_seed(777)         # 동일한 재현성을 위하여, 시드 동일하게 초기화

xy = np.genfromtxt(CSV_DIR + FNAME, delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]    # [n, [0~-2]] ..  [처음~(n-1)열] = [n,3]
y_data = xy[:, [-1]]    # [n, [-1]]  ...  [마지막열] = [n,1]

# print(len(x_data[0]))   # 행(n) = 데이터셋의 갯수
# print(len(x_data[1]))   # nb_classes = 입력변수의 갯수
nb_classes = len(x_data[1])


""" Make sure the shape and data are OK """
# print("matrics=%s  ....  [n,3] \n%s\n" % (x_data.shape, x_data))
# print("matrics=%s  ....  [n,1] \n%s" % (y_data.shape, y_data))


""" placeholder for a tensor that will be always fed. """
# 독립변수(X), 종속변수(Y), 가중치(W), 편향(b)
# 독립/종속변수 = 플레이스홀더 (자리만 예약 해 둠--피딩으로 공급)
X = tf.placeholder(tf.float32, shape=[None, nb_classes])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# 가중치(W)와 편향(b)는 고정값으로 저장된다.
W = tf.Variable(tf.random_normal([nb_classes,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


""" hypothesis : 예측값(H) = W.X + b """
# minimize cost() ... alpha = learning_rate(tiny step = 1e-5)
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))     # R**2
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
train = optimizer.minimize(cost)


""" draw graph in session """
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

""" Data solving = Training """
for step in range(REPEATATION):
    cost_val, hypo_val, _ = sess.run(
        [cost, hypothesis, train],              # C[H(x),y]= w=minimize_cost()
        feed_dict={X:x_data, Y:y_data})

    if step%500 == 0:
        print("{:0>5,}___cost:{:5.2f}".format(
            step,
            cost_val,))

""" 최종 학습값 확인 """
fW = W.eval(session=sess)
fb = b.eval(session=sess)

print('\n\n')
print("\n최종 Cost :", cost_val)
# print(hypo_val)
print("\n가중치(W)매트릭스 :\n", fW)
print("\n편차(b) :", fb)


# 76,83,71,149
# 96,93,95,192
# x1_data = [[100, 70, 101]]
# x2_data = [[60, 70, 110]]
x1_data = [[76, 83, 71]]
x2_data = [[96, 93, 95]]

# H(x) = x:[1,3]x W:[3,1] = [1,1].... 각각 입력 할 경우 (n=1)
prediction_01 = sess.run(hypothesis, feed_dict= {X: x1_data})
prediction_02 = sess.run(hypothesis, feed_dict= {X: x2_data})

print("\n\n--------------------------------------")
print("  When your scores were  {:}".format(x1_data))
print("  Your  score will be..  {:}".format(prediction_01))

print("\n\n--------------------------------------")
print("  When other scores were {:}".format(x2_data))
print("  Other score will be..  {:}".format(prediction_02))

print('\n\n')
res1 = np.add(np.matmul(x1_data, fW),fb)
res2 = np.add(np.matmul(x2_data, fW),fb)
print(res1)
print(res2)


# 엑셀 분석값과 비교
x1_data = [[76], [83], [71]]    # 라벨값 y = 149
x2_data = [[96], [93], [95]]    # 라벨값 y = 192
eW = [1.9871, 1.7873, 1.8525]   # 엑셀의 w1, w2, w3 .. R2=
eb = [0.1336, 19.982, 15.622]   # 엑셀의 b1, b2, b3

res_mat = np.add(np.dot(eW, x1_data), eb)
print("\n\n%s  ... 라벨값 y1 = 149 \n%s \n%s" % (res_mat.shape, res_mat/3, np.mean(res_mat)))
print("첫번째 인자로 만 계산할 경우(94.54%) :", x1_data[0][0] * eW[0] + eb[0])

res_mat = np.add(np.dot(eW, x2_data), eb)
print("\n\n%s  ... 라벨값 y2 = 192 \n%s \n%s" % (res_mat.shape, res_mat/3, np.mean(res_mat)))
print("첫번째 인자로 만 계산할 경우(94.54%) :", x2_data[0][0] * eW[0] + eb[0])
