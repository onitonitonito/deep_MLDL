""" multi-variable regression
# 여러 개의 입력(feature)의 Linear Regression
# 소프트맥스 회귀 (SoftMAx regression)
# ------------
# hypothesis = x1*w1 + x2*w2 + x3*w3 + b
#   = H(x1, x2, x3) = X* W -->
#   = [1x3] = [1,3]*[3,1] = [1x1]
# [케글:Kaggle] = https://www.kaggle.com --> 데이터과학 예제 무료로 연습하는 법.
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

CSV_DIR = os.path.join(os.path.dirname(__file__), "_csv_hunkim/")

def single_linear_regression(learnig_rate=1e-5, repeatation=2001):
    """ 전통적인 방법의 선형회귀(Linear Regression)
    # 3개의 과목에 대하여, 5명의 성적와 최종득점에 대한 예측치를 구한다.
    # x1, 2, 3 = 3과목 : 주어지는 값의 종류 -- 독립변수(X)
    # y값(종속변수) ... 독립변수(x)에 관한 선형관계다 (선형회귀)
    #   - y = H(x) = W(가중치).x + b(편차) + ε(오차)
    """
    x1_data = [73., 93., 89., 96., 73.]
    x2_data = [80., 88., 91., 91., 66.]
    x3_data = [75., 93., 90., 100., 70.]
    y_data = [152., 185., 180., 196., 142.]

    """ placeholder for a tensor that will be always fed
    # 플래이스홀더도 자료의 형식을 지정해 준다. float32
    # 가중치(w)를 찾아낸다 -- 변수형식
    """
    x1 = tf.placeholder(tf.float32)
    x2 = tf.placeholder(tf.float32)
    x3 = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    w1 = tf.Variable(tf.random_normal([1]), name ='weight1')
    w2 = tf.Variable(tf.random_normal([1]), name ='weight1')
    w3 = tf.Variable(tf.random_normal([1]), name ='weight1')
    b = tf.Variable(tf.random_normal([1]), name ='bias')

    hypothesis = x1*w1 + x2*w2 + x3*w3 + b      # H(x) = X*W + b

    """ cost/loss function
    # 코스트 함수는 (가설-실제)**2의 전체평균값 = (1/m*Sum[ H(x)-y)^2 ]
    # 가설값, H(x) = Wx + b, y=실제 산출된 값 (과목의 총점)
    """
    cost = tf.reduce_mean(tf.square(hypothesis - y))


    """ loss 의 최소화 = 코스트함수의 최소화 : minimizeCose()
    # Need a very small learning rate(=α) for this data set
    # 이산 미소값 알파(α)만큼, 코스트함수의 미분값이 0이 될때까지
    # 경사면(W)를 타고 내려간다 = Gradient Descent Optimizer(알고리즘)
    # W: = W - alpha * (roe / roe_W) * cost(W) ... b를 생략한 값: cost(W,b)
    # 러닝비율(알파)은 10^-5 보다 작아야, 프로그램이 돌아가는 것 (확인)
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learnig_rate)
    train = optimizer.minimize(cost)

    """ Launch the graph in a session """
    sess = tf.Session()

    """ Initialize global variables in the graph """
    sess.run(tf.global_variables_initializer())

    for step in range(repeatation):
        cost_val, hy_val, _ = sess.run(
            [cost, hypothesis, train],
            feed_dict = {x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})

        if step % 100  == 0:    # 10의 배수에서 중간값을 확인한다.
            # print(step, "Cost= ", cost_val, "\nPrediction= ", hy_val)
            print("\n\n{:0>5,}__Cost:{:5.2f} ______ Prediction = \n{:}".format(
                step,           # 2,000 미소편차 2천번 반복
                cost_val,       # 코스트함수는 가설과 이론의 편차 --> Loss최소화
                hy_val))        # 예측값(y_)을 출력한다...np.ndarray클래스
    print("\n\nType of hypothesis = ", type(hy_val))
# single_linear_regression()

def matrics_linear_regression(learnig_rate=1e-5, repeatation=2001):
    """ H(X) = X* W ..... x:[5,3] x w:[3,1] = y=H(x):[5x1]
    # 3과목, 자료의 갯수(n)개 일때, --> x:[n,3] x w:[3,1] = y:[n,1]
    """
    x_data = [
        [73., 80., 75.],
        [93., 88., 93.],
        [89., 91., 90.],
        [96., 91., 100.],
        [73., 66., 70.]]

    y_data = [
        [152.],
        [185.],
        [180.],
        [196.],
        [142.]]

    """ placeholder for a tensor that will be always fed. """
    X = tf.placeholder(tf.float32, shape= [None, 3])    # None = 'n'
    Y = tf.placeholder(tf.float32, shape= [None, 1])    # None = 'n'

    W = tf.Variable(tf.random_normal([3,1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    """ hypothesis : H(X) = X*W + b """
    hypothesis = tf.matmul(X, W) + b
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learnig_rate)
    train = optimizer.minimize(cost)

    """ Launch the graph in a session """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())     # 그래프의 세션변수를 초기화

    for step in range(repeatation):           # 2000번 미소스텝반복, 타고내려온다.
        cost_val, hypo_val, _ = sess.run(
            [cost, hypothesis, train],
            feed_dict={X: x_data, Y: y_data})
        if step%100 == 0:                   # 100의 배수마다 중간값 점검
            print("\n{:0>5,}__cost:{:5.2f} ____ Prediction = \n {:}".format(
                step,
                cost_val,
                hypo_val))
# matrics_linear_regression()

def show_each_prediction_01_02(sess, hypothesis, X):
    """ ASK my score prediction : 한 사람의 예측값을 알고 싶다... helper()
    # 한사람의 결과만 묻기 때문에, 자료의 갯수(n=1)
    # ... H(x) = X:[1,3] x W:[3,1] = y_:[1,1] 한 가지 예측값이 나온다.
    # 인자 값 = 계산을 위해서 정의되지 않은 값은 외부에서 받아온다
    #    - sess = tf.Session()
    #    - hypothesis = X.W + b
    #    - X : x_data = xy[:, 0:-1] from CSV_DIR = "./_csv_hunkim/"+ data.CSV
    """
    x1_data = [[100, 70, 101]]
    x2_data = [[60, 70, 110]]

    # H(x) = x:[1,3]x W:[3,1] = [1,1].... 각각 입력 할 경우 (n=1)
    prediction_01 = sess.run(hypothesis, feed_dict= {X: x1_data})
    prediction_02 = sess.run(hypothesis, feed_dict= {X: x2_data})

    print("\n\n--------------------------------------")
    print("  When your scores were  {:}".format(x1_data))
    print("  Your  score will be..  {:}".format(prediction_01))

    print("\n\n--------------------------------------")
    print("  When other scores were {:}".format(x2_data))
    print("  Other score will be..  {:}".format(prediction_02))

def matrics_file_import_linear(learnig_rate=1e-5, repeatation=2001):
    """ Lecture.04-2 : 파일로 매트릭스 데이터 읽어오기
    # CSV화일 임포트, 매트릭스 선형회귀 (metrics linear regression)
    # 임포트 화일 = CSV_DIR+ 'data01_test_score.csv'
    # CSV에 저장된 값은 [n,3]x[3,1] = [n,1] 의 매트릭스 배열
    """
    tf.set_random_seed(777)         # for reproducibility
    xy = np.loadtxt(
        CSV_DIR+ "data01_test_score.csv",
        delimiter=',',
        dtype=tf.float32)

    x_data = xy[:, 0:-1]    # 행=모든것, 열= 처음~마지막열의 앞까지 (n-1)
    y_data = xy[:, [-1]]    # 행=모든것, 열= 마지막열(n)

    """ Make sure the shape and data are OK """
    print("matrics={:} .... (n,3) \n{:}".format(
        x_data.shape, x_data))

    print("\n\nmatrics={:} .... (n,1) \n{:}".format(
        y_data.shape, y_data))

    """ placeholder for a tensor that will be always fed. """
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([3,1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    """ hypothesis """
    hypothesis = tf.matmul(X, W) + b

    """ minimize cost() ... alpha = learning_rate(tiny step = 1e-5) """
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    Optimizer = tf.train.GradientDescentOptimizer(learning_rate=learnig_rate)
    train = Optimizer.minimize(cost)

    """ draw graph in session """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(repeatation):
        cost_val, hypo_val, _ = sess.run(
            [cost, hypothesis, train],              # C[H(x),y]= w=minimize_cost()
            feed_dict={X:x_data, Y:y_data})
        if step%100 == 0:
            print("\n-----------------------------------")
            print("{:0>5,}___cost:{:5.2f} ___ Prediction = \n{:}".format(
                step,
                cost_val,
                hypo_val))

    show_each_prediction_01_02(sess, hypothesis, X)

def thread_queues_runner_linear(learnig_rate=1e-5, repeatation=2001):
    """ 여러 개 화일CSV를 순차적으로 Que에 넣고, 학습하는 Batch prediction
    # 메모리에 한꺼번에 올릴수 없음 = 큐 러너 시스템 (Queue Runners System)
    # (1) 화일네임들 -> 랜덤셔플 -> 화일네임큐(queue)에 쌓는다.
    # (2) 화일네임큐 -> 리더1,2... -> 디코드1,2.. -> 샘플링큐(queue)에 쌓는다.
    # (3) 샘플링 큐 --> 하나씩 배치(Batch) 프로세스 실행
    # ------
    #  - 화일네임큐 = tf.train.slice_input_producer([화일1, 화일2, 화일3.csv..],
    #         셔플=트루, 네임=화일네임큐).. 셔플로 섞어서 화일네임 큐에 쌓은다.
    #  - 리더 = tf.TextLineReader() / _, val = reader.read(화일네임큐)
    #  - record_defaults = [[0.],[0.],...]
    #    xy = tf.decode_csv(value, record_defaults=record_defaults )
    """
    tf.set_random_seed(777)  # for reproducibility
    filename_queue = tf.train.string_input_producer(
        [CSV_DIR+ 'data01_test_score.csv'], name='filename_queue')

    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)

    """ default value, in case of empty columns,
    # also specifies the type of the decorate result
    """
    record_defaults = [[0.], [0.], [0.], [0.]]
    xy = tf.decode_csv(value, record_defaults=record_defaults)

    """ collect batches of csv in """
    train_x_batch, train_y_batch = tf.train.batch(
        [xy[0:-1], xy[-1:]],
        batch_size=10)

    """ placeholder for a tensor that will be always fed. """
    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([3,1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.matmul(X, W) + b

    """ simplified cost / loss function = minimizeCost() """
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learnig_rate)
    train = optimizer.minimize(cost)

    # sess = tf.Session()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        """ Batch로 인해서, 추가되는 부분 """
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(repeatation):
            x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
            cost_val, hypo_val, _ = sess.run(
                [cost, hypothesis, train],
                feed_dict={X: x_batch, Y: y_batch})
            if step%100 == 0:
                # print("{:0>5}__cost: {:} _____ prediction: \n{:}".format(
                #     step, cost_val, hypo_val))
                print("{:>5}__cost: {:}".format(
                    step, cost_val))
        coord.request_stop()
        coord.join(threads)

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
    # 소프트맥스() 만 적용한 상태, 'Hot-dot Encoding' 적용 전..
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

            if step%10 == 0:
                print("{:0>5,}__cost= {:}".format(
                    step,
                    sess.run(cost, feed_dict={X:x_data , Y:y_data}))
                    )

            """ for drawing cost_minimize : cost_plots """
            cost_plots.append(
                sess.run(
                    cost,
                    feed_dict={X:x_data , Y:y_data}))

        """ draw cost_minimize curve graph """
        draw_lists_pyplot(cost_plots, line_weight=0.5, learnig_rate=learnig_rate)

        separator = "----------------------"
        print("\n\nHypothesis= \n{:}\n{:}......\n\n".format(
            separator,
            sess.run(
                hypothesis,
                feed_dict={X:x_data , Y:y_data})))

        """ ASK PREDICTION """
        show_asking_result_to_softmax(
            sess,
            hypothesis,
            X,
            [[1, 11, 7, 9]])

        show_asking_result_to_softmax(
            sess,
            hypothesis,
            X,
            [
                [1, 11, 7, 9],
                [1, 3, 4, 3],
                [1, 1, 0, 1]])

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

if __name__ == '__main__':
    # single_linear_regression(4.5e-5, 2001)         # cost=0.36
    # matrics_linear_regression(1e-5, 2001)          # cost=0.43
    # matrics_file_import_linear(1e-5, 2001)         # cost=0.87
    thread_queues_runner_linear(1e-5, 10001)        # cost = 4.55 -> 4.42 min.
    # softMax_func_to_probablity(0.75, 2001)          # cost=0.0012
    # softMax_func_to_probablity(0.11, 1301)          # cost=0.0012
