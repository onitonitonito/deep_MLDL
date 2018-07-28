"""
# 자동완성 추천 RNN 학습하기
#-----------------------------------
# 자연어 처리나 음성처리 분야에 많이 사용되는 RNN의 기본적인 사용법을 익혀 봄.
# 4글자를 가진 단어를 학습시켜, 3글자만 주어지면, 나머지 1글자는 추천하여 단어를
# 완성하는 학습과정, 각 글자가 나타나는 시퀀스를 판단하여, 다음글자가 무엇이
# 와야 되는지를 판단하는 연습, 중복문제는 없으므로, 순수하게 순서패턴에 의해서
# 학습을 한다, 응용하면 - 단어조합의 순서로 추천해 줌
#       - {치킨}에는 {맥주}, {피자}에는 {콜라}
#       - 피자 다음 나타날 음료의 확률은 {맥주}보다는 {콜라}가 높다.
#
#"""
print(__doc__)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_dic = { cha : i for i, cha in enumerate('abcdefghijklmnopqrstuvwxyz')}
chars = list(num_dic.keys())
dic_len = len(num_dic)          # 26 개의 알파벳 글자

seq_data = [
        'word', 'wood', 'deep', 'dive', 'cold',
        'cool', 'load', 'love', 'kiss', 'kind',
        'this', 'that', 'then', 'toss', 'tang',
        'many', 'mass', 'moss', 'mole', 'mist',
        'tail', 'toll', 'test', 'tone', 'tear',
        'boss', 'bear', 'boot', 'bone', 'bits',
        'beer', 'cost', 'dust', 'doll', 'dead',
        'pear', 'bore', 'post', 'poll', 'pill'
    ]

n_step = 3                     # 타입스텝: [1 2 3] => 3 ... 시퀀스의 갯수.
n_input = n_class = dic_len    # 알파벳 26글자
n_hidden = 128                 # 히든 레이어의 갯수 = 128개
learning_rate = 0.01           # 학습률 0.01 에서 가장 학습효과가 좋음.
total_epoch = 100


def make_batch(seq_data):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[:-1]]
        target = num_dic[seq[-1]]
        input_batch.append(np.eye(dic_len)[input])
        target_batch.append(target)

    return input_batch, target_batch

# input_batch         # [3,26] ... 2차원 리스트
input_batch, target_batch = make_batch(seq_data)

""" Y = WX + b [None] = [None, 3, 26].[128, 26] + [26] """
X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.int32, [None])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))  # [128, 26]
b = tf.Variable(tf.random_normal([n_class]))            # [128]

"""
# 기본 RNN 셀(cell1)을 생성 = Basic LSTM (Long Short-Term Memory)
# 과적합 방지를 위한 Dropout 기법을 사용 ... 0.5 드롭아웃
# 여러 개의 셀을 조합하기 위해 RNN 셀을 추가(cell2)로 생성
# 여러 개의 셀을 조합한 RNN 셀(multi_cell = cell1 + cell2)을 생성
# tf.nn.dynamic_rnn 함수를 이용, 순환(dynamic)신경망 형성한다
# 최종 결과는 one-hot 인코딩 형식으로 만듭니다
"""

cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)

cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#########
# 신경망 모델 학습
#########

sess = tf.Session()
sess.run(tf.global_variables_initializer())

dots = []
for epoch in range(total_epoch):
    _, loss = sess.run(
            [optimizer, cost],
            feed_dict={X: input_batch, Y: target_batch},
        )
    dots.append(loss)
    print('Epoch:', '%04d' % (epoch + 1),
         'cost =', '{:.6f}'.format(loss))


print('최적화 완료!, cost = {:.2f} %'.format(loss*100))

plt.title('Learning rate - cost = {:.2f} %'.format(loss*100))
plt.plot(dots)
plt.ylabel('Cost')
plt.show()


#########
# 결과 확인
#########
# 레이블값이 정수이므로 예측값도 정수로 변경해줍니다.
# one-hot 인코딩이 아니므로 입력값을 그대로 비교합니다.

prediction = tf.cast(tf.argmax(model, 1), tf.int32)
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

predict, accuracy_val = sess.run(
        [prediction, accuracy],
        feed_dict={X: input_batch, Y: target_batch}
    )

predict_words = []
for idx, val in enumerate(seq_data):
    last_char = chars[predict[idx]]
    predict_words.append(val[:3] + last_char)

bools = [predict_words[i] == seq_data[i] for i, _ in enumerate(chars)]

print('\n=== 예측 결과 ===')

for i, word in enumerate(seq_data):
    _input = word[:3] + "?"
    _predic = predict_words[i]
    _answer = word

    line = "Q.{:<2}: {} = Pred: {} ({}) ... {}".format(
        i+1,
        _input,
        _predic,
        _answer,
        _predic == _answer,
        )
    print(line)

print('\n정확도:', accuracy_val)
