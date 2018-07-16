"""
# RNN Text Generation Model --Solaris ML-Lab
# http://solarisailab.com/archives/2487
#
# 머신러닝 RNN을 이용한 텍스트 제네레이터
#------------------------------
# Author : solaris33
# Project URL : http://solarisailab.com/archives/2487
# GitHub Repository : https://github.com/solaris33/char-rnn-tensorflow/
# Reference : https://github.com/sherjilozair/char-rnn-tensorflow
#\n\n\n"""
print(__doc__)

import tensorflow as tf
import numpy as np

from utils import TextLoader

# 학습에 필요한 설정값들을 지정합니다.
data_dir = '_rnn_data'

#data_dir = 'data/linux'
batch_size = 50 # Training : 50, Sampling : 1
seq_length = 50 # Training : 50, Sampling : 1
hidden_size = 128   # 히든 레이어의 노드 개수
learning_rate = 0.002
num_epochs = 2
num_hidden_layers = 2
grad_clip = 5   # Gradient Clipping에 사용할 임계값

# TextLoader를 이용해서 데이터를 불러옵니다.
data_loader = TextLoader(data_dir, batch_size, seq_length)
# 학습데이터에 포함된 모든 단어들을 나타내는 변수인 chars와 chars에 id를 부여해 dict 형태로 만든 vocab을 선언합니다.
chars = data_loader.chars
vocab = data_loader.vocab
vocab_size = data_loader.vocab_size # 전체 단어개수

# 인풋데이터와 타겟데이터, 배치 사이즈를 입력받기 위한 플레이스홀더를 설정합니다.
input_data = tf.placeholder(tf.int32, shape=[None, None])  # input_data : [batch_size, seq_length])
target_data = tf.placeholder(tf.int32, shape=[None, None]) # target_data : [batch_size, seq_length])
state_batch_size = tf.placeholder(tf.int32, shape=[])      # Training : 50, Sampling : 1

# RNN의 마지막 히든레이어의 출력을 소프트맥스 출력값으로 변환해주기 위한 변수들을 선언합니다.
# hidden_size -> vocab_size
softmax_w = tf.Variable(tf.random_normal(shape=[hidden_size, vocab_size]), dtype=tf.float32)
softmax_b = tf.Variable(tf.random_normal(shape=[vocab_size]), dtype=tf.float32)

# num_hidden_layers만큼 LSTM cell(히든레이어)를 선언합니다.
cells = []
for _ in range(0, num_hidden_layers):
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cells.append(cell)

# cell을 종합해서 RNN을 정의합니다.
cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

# 인풋데이터를 변환하기 위한 Embedding Matrix를 선언합니다.
# hidden_size -> vocab_size
embedding = tf.Variable(tf.random_normal(shape=[vocab_size, hidden_size]), dtype=tf.float32)
inputs = tf.nn.embedding_lookup(embedding, input_data)

# 초기 state 값을 0으로 초기화합니다.
initial_state = cell.zero_state(state_batch_size, tf.float32)

# 학습을 위한 tf.nn.dynamic_rnn을 선언합니다.
# outputs : [batch_size, seq_length, hidden_size]
outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, dtype=tf.float32)
# ouputs을 [batch_size * seq_length, hidden_size]] 형태로 바꿉니다.
output = tf.reshape(outputs, [-1, hidden_size])

# 최종 출력값을 설정합니다.
# logits : [batch_size * seq_length, vocab_size]
logits = tf.matmul(output, softmax_w) + softmax_b
probs = tf.nn.softmax(logits)

# Cross Entropy 손실 함수를 정의합니다.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target_data))

# 옵티마이저를 선언하고 옵티마이저에 Gradient Clipping을 적용합니다.
# grad_clip(=5)보다 큰 Gradient를 5로 Clippin합니다.
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.apply_gradients(zip(grads, tvars))

# 세션을 열고 학습을 진행합니다.
with tf.Session() as sess:
    # 변수들에 초기값을 할당합니다.
    sess.run(tf.global_variables_initializer())

    for e in range(num_epochs):
        data_loader.reset_batch_pointer()
        # 초기 상태값을 지정합니다.
        state = sess.run(initial_state, feed_dict={state_batch_size : batch_size})

        for b in range(data_loader.num_batches):
            # x, y 데이터를 불러옵니다.
            x, y = data_loader.next_batch()
            # y에 one_hot 인코딩을 적용합니다.
            y = tf.one_hot(y, vocab_size)            # y : [batch_size, seq_length, vocab_size]
            y = tf.reshape(y, [-1, vocab_size])       # y : [batch_size * seq_length, vocab_size]
            y = y.eval()

            # feed-dict에 사용할 값들과 LSTM 초기 cell state(feed_dict[c])값과 hidden layer 출력값(feed_dict[h])을 지정합니다.
            feed_dict = {input_data : x, target_data: y, state_batch_size : batch_size}
            for i, (c, h) in enumerate(initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

            # 한스텝 학습을 진행합니다.
            _, loss_print, state = sess.run([train_step, loss, final_state], feed_dict=feed_dict)

            print("{}(학습한 배치개수)/{}(학습할 배치개수), 반복(epoch): {}, 손실함수(loss): {:.3f}".format(
                          e * data_loader.num_batches + b,
                          num_epochs * data_loader.num_batches,
                          (e+1),
                          loss_print))

    print("트레이닝이 끝났습니다!")


    # 샘플링 시작
    print("샘플링을 시작합니다!")
    num_sampling = 4000  # 생성할 글자(Character)의 개수를 지정합니다.
    prime = u' '         # 시작 글자를 ' '(공백)으로 지정합니다.
    sampling_type = 1    # 샘플링 타입을 설정합니다.
    state = sess.run(cell.zero_state(1, tf.float32)) # RNN의 최초 state값을 0으로 초기화합니다.

    # Random Sampling을 위한 weighted_pick 함수를 정의합니다.
    def weighted_pick(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        return(int(np.searchsorted(t, np.random.rand(1)*s)))

    ret = prime       # 샘플링 결과를 리턴받을 ret 변수에 첫번째 글자를 할당합니다.
    char = prime      # Char-RNN의 첫번쨰 인풋을 지정합니다.
    for n in range(num_sampling):
        x = np.zeros((1, 1))
        x[0, 0] = vocab[char]

        # RNN을 한스텝 실행하고 Softmax 행렬을 리턴으로 받습니다.
        feed_dict = {input_data: x, state_batch_size : 1, initial_state: state}
        [probs_result, state] = sess.run([probs, final_state], feed_dict=feed_dict)

        # 불필요한 차원을 제거합니다.
        # probs_result : (1,65) -> p : (65)
        p = np.squeeze(probs_result)

        # 샘플링 타입에 따라 3가지 종류로 샘플링 합니다.
        # sampling_type : 0 -> 다음 글자를 예측할때 항상 argmax를 사용
        # sampling_type : 1(defualt) -> 다음 글자를 예측할때 항상 random sampling을 사용
        # sampling_type : 2 -> 다음 글자를 예측할때 이전 글자가 ' '(공백)이면 random sampling, 그렇지 않을 경우 argmax를 사용
        if sampling_type == 0:
            sample = np.argmax(p)
        elif sampling_type == 2:
            if char == ' ':
                sample = weighted_pick(p)
            else:
                sample = np.argmax(p)
        else:
            sample = weighted_pick(p)

        pred = chars[sample]
        ret += pred     # 샘플링 결과에 현재 스텝에서 예측한 글자를 추가합니다. (예를들어 pred=L일 경우, ret = HEL -> HELL)
        char = pred     # 예측한 글자를 다음 RNN의 인풋으로 사용합니다.

    print("샘플링 결과:")
    print(ret)


"""
921(학습한 배치개수)/924(학습할 배치개수), 반복(epoch): 2, 손실함수(loss): 1.522
922(학습한 배치개수)/924(학습할 배치개수), 반복(epoch): 2, 손실함수(loss): 1.459
923(학습한 배치개수)/924(학습할 배치개수), 반복(epoch): 2, 손실함수(loss): 1.507
트레이닝이 끝났습니다!
샘플링을 시작합니다!
샘플링 결과:
 mad.

First God EDWARD:
Speak peace; one it sunce his come, so fix then.
Where I see him bucked or kings this queen him,
Hake him, our deforth.

BIANA:
No beather, our me? I drub! I will not
Instruns: I do do stillistrotrand;
I Jintoly, he ftratius against
It will never as our beody him,
as the leaved of with so?

AUTOLYCUS:
Why, gleasis'd it a king, sworrd an ear
too, lord, Musiciansle, omans, no parchiof
Hath so put souns;
And I could in stides
Have nease me mie Yotur; but sterms him Is.
Or, were rettlempice, theren, you must, a hoome hatwain,
And that thee, shall commen armmford, means,
I needle the drows him of other become well.

ANTONIO:
And see, amder him, expecimed of.

POLICENTIO:
Shall deer to your man on thing?
Whil; louting, hold defen bold? strake out livito herely brains,
Is in sure a heart
are ever.

TRANIO:
Come, for thee to but together assion is some the
the gated away not against them,-nato stersite
That wit bey dead foes dut delading:
And, gracious shall be a liable the
uschurses duld; I recea, he
Why ass Henry ,ike wrop him wathis potinesss,
Even 'ex grial of him but helds and fare: oather's name.

GEMNIUH:
Un thine, she, for upon years.

SaBEBASTIA:
What it is it, On anutors by his for fight me;
He daubtter feell Ruter, for our napilib.
We here heard in unlockstluse hece
By those renhour here darll, life so could remberam shot!
Put loved me no man? and he have down:
I, if but of me
Frant, broke you vengy to-drown, whither opass,
That no two so; and in have!

BAPTISTA:
Hereford? Hurst! Rice, the greed back.

First Murderen:
You present dam, guil! usir, and my twounders
Agess counself to your have at speaks;
Wherey to the stover of you heart firs,
What spraceing her, I have him, brother's jack grawls,
The there of the viila,
Unle a MurFon frawn'dsreit words; and so more
There both to this.

PRINA:
Go sim; take mar, then, may let she bide of
O, inken Voldantio; that cheilow's frover'd
Unband my prehas victio!

First, in my eye rever: be not the farget
That repart charge the marry Richer Cit
Why with make of Hartip, and prince by Monembras,
And tell! think hom to gentle for whyl come, nock
Harking you though him creakseg in him agoran fougint,
Ele to your pluchious mightio.
I chaptle a friance nowness; may hand:
If it bet ame and back of Barking aim
Ment of dowbmen in actime a man,
In will we shill beleg maderas
Unlock, More's begst thing he aforthy, or I shall up?
Of play, all and revenge held grace, his good in
for dead not; might will live
Ancess, Barkingh that,
And you thou, privouties of of what.
What thou womaws, have be somedion. When, budish
his at thrower, trock in the please a macheak?
My lord.

FERTIUS:
By his bitter.

DUCHESS OFY LARENCE:
Why, Senators, be bonam.

POLIXENES:
Manish, I Pad knew us not it
Wisoria mone upon this paparess,
And name! paur gatul nighten, and thou will Warwike shows
Think armment we
Are isberor charg to towerge for king sire!
Shalzer has faceful pecutioutae!

SLORENTE:
Against, I'll kink the father flatter
For her die igniollibair, Rutles
are a she mightife; and show them, should: Englard.
One Gquiciues,
So beat secred that rema! for him!
We like me same, thou may, sir, help dost.

BUtHAS:
More not have unfling by fair in think
And shall as know so,-whot make not
Here cempia,
And sake us to your love! like a doubt-reass.
Hath but down him! should shart my cate gate doth.

Pedant:
He damgied the way Cango,
Romery gold miscalan and strpepardlainess
This name?

Clown:
If I part a gentle your like fingen's a deas
He will ditch in him, like dowell say,
A for this getter.

KING RICHARD III:
Artious of a cluste? them!
Where thank fleens since so be degied;'
Thou art for against the were poror man; Camilarm,
go double forgunt, her not since
For tresy! I'll I pardon for draw nor lie;
Witharof he hap a me, forth against-tank shall flass,
I dights

Process returned 0 (0x0)        execution time : 36521.034 s
계속하려면 아무 키나 누르십시오 . . .

"""
