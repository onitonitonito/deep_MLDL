# 구글.콜랩 주피터.ipynb 화일(py)
# 텐서플로우 모듈 설치
# !pip install tensorflow

# For the current version:
# !pip install --upgrade tensorflow

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def show_asking_result_to_softmax(sess, hypothesis, X, x_data):
        # testing & 'One-Hot' Encoding
        predict = sess.run(
            hypothesis,
            feed_dict={X:x_data})

        print("\n\n%s \n___ 'One-Hot'Encoding choice = %s" %(
            predict,
            sess.run(tf.arg_max(predict, 1))))    # 'One-Hot'Encoding

def softMax_func_to_probablity(learnig_rate=1e-5, repeatation=2001):
    """ X.W = Y ...  X:[nx4] * W:[4x3] + b:[3] = Y:[nx3]
    # .. n=8(행) / nb_classes = 3(열) ... 예측값 Y: [n x nb_classes]
    # 소프트맥스()만 적용한 상태, 'Hot-dot Encoding' 적용 전..
    """
    x_data = [      # [8x4] 행열
        [1, 2, 1, 1],
        [2, 1, 3, 2],
        [3, 1, 3, 4],
        [4, 1, 5, 5],
        [1, 7, 5, 5],
        [1, 2, 5, 6],
        [1, 6, 6, 6],
        [1, 7, 7, 7]]

    y_data = [      # Y:[8x3] 행열   - 'One-hot' Encoding 표기법
        [0, 0, 1],    # 2  ... 레이블의 갯수 = 3 = number_classes --> 0,1,2
        [0, 0, 1],    # 2
        [0, 0, 1],    # 2
        [0, 1, 0],    # 1
        [0, 1, 0],    # 1
        [0, 1, 0],    # 1
        [1, 0, 0],    # 0
        [1, 0, 0]]    # 0

    """ placeholder = x:[n,4] * w:[4,3] + b:[3] = y:[n,3] """
    X = tf.placeholder(dtype="float32", shape=[None, 4])
    Y = tf.placeholder(dtype="float32", shape=[None, 3])
    nb_classes = 3      # columns (예측값의 열 = 편향값의 행(?)  )

    W = tf.Variable(tf.random_normal([4, nb_classes], name='weight'))
    b = tf.Variable(tf.random_normal([nb_classes], name='bias'))

    """ tf.nn.softmax computes softmax activations
    # softmax = exp(logits) / reduce_sum(exp (logits), dim)
    """
    hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

    """ Cross Entropy cost/Loss """
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
    train = tf.train.GradientDescentOptimizer(learning_rate=learnig_rate)
    # learning_rate = 0.81... 에서 minimize_cost()거의 '0'에 도달

    optimizer = train.minimize(cost)

    """ Launch graph """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        cost_plots = []
        for step in range(repeatation):
            sess.run(
                optimizer,
                feed_dict={X:x_data , Y:y_data})

            # if step%10 == 0:
            #     print("{:0>5,}__cost= {:}".format(
            #         step,
            #         sess.run(cost, feed_dict={X:x_data , Y:y_data}))
            #         )

            """ for drawing cost_minimize : cost_plots """
            cost_plots.append(
                sess.run(
                    cost,
                    feed_dict={X:x_data , Y:y_data}))

        separator = "----------------------"
        # print("\n\nHypothesis= \n{:}\n{:}......\n\n".format(
        #     separator,
        #     sess.run(
        #         hypothesis,
        #         feed_dict={X:x_data , Y:y_data})))
        # """ ASK PREDICTION """
        # show_asking_result_to_softmax(
        #     sess,
        #     hypothesis,
        #     X,
        #     [[1, 11, 7, 9]])
        # show_asking_result_to_softmax(
        #     sess,
        #     hypothesis,
        #     X,
        #     [
        #         [1, 11, 7, 9],
        #         [1, 3, 4, 3],
        #         [1, 1, 0, 1]])
    return cost_plots

def draw_lists_pyplot(y_array, line_weight=3, learnig_rate=1):
    """ Basic Array Plotting Graph """
    y = y_array
    plt.plot(y, lw=line_weight, label='cost(a={:})'.format(learnig_rate))
    plt.legend()

    plt.title("Gradient Descent Optimizing Method\nminimize cost function")
    plt.xlabel('time-itoration')
    plt.ylabel('cost-function')
    plt.xlim(0,)
    plt.ylim(0,)

    plt.grid(b=None, which='major', axis='both')
    plt.show()

"""# 미소스텝 알파(러닝비율)과 이터레이션 크기와 관계 (과정Graph)
### Learning Rate(α) 값을 절반으로 줄이는 대신, iteration 을 2배씩 늘리면서
### Gradient Descent Method (Cross-Entropy를 이용해서 minimize cost)
"""

# %matplotlib inline

def draw_minimize(learning_rate=0.1, repeat=2001):
    cost_plots = softMax_func_to_probablity(learning_rate, repeat)
    draw_lists_pyplot(cost_plots, 0.3, learning_rate)
    print("LAST_COST_VALUE = ", cost_plots[-1])

# draw_minimize(1, 501)       # 에러 = nan
draw_minimize(0.8, 1001)
draw_minimize(0.4, 2001)
draw_minimize(0.2, 4001)
draw_minimize(0.1, 8001)
draw_minimize(0.05, 16001)
# draw_minimize(0.025, 32001)

"""#결과 값은,  최소 Cost()는 약 0.04 ~ 0.05 부근,
# > * Learning_rate 은 약 0.1정도가 가장 효율적.... 인 것 같음..
2018-01-31 11:53:41.924584: I C:\tf_jenkins\home\workspace\rel-win\M\windows\
PY\36\tensorflow\core\platform\cpu_feature_guard.cc:137] Your CPU supports
instructions that this TensorFlow binary was not compiled to use: AVX AVX2

LAST_COST_VALUE =  0.00337092
LAST_COST_VALUE =  0.0136384
LAST_COST_VALUE =  0.0396193
LAST_COST_VALUE =  0.0498026
LAST_COST_VALUE =  0.0502757

Process returned 0 (0x0)        execution time : 39.630 s
"""
