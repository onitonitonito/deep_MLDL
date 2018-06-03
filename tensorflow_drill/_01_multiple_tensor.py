import tensorflow as tf

a = tf.constant(value=20, dtype=tf.float32, shape=[], name='a')
b = tf.constant(value=30, dtype=tf.float32, shape=[], name='b')
mulop = a * b

# 세션생성하기
sess = tf.Session()

# 텐서보드 사용하기
# tw = tf.train.SummaryWriter() 2016년11/30일 - 변경되었음.
tw = tf.summary.FileWriter('./_logdir/', graph=sess.graph)


# 세션 실행하기
print(sess.run(mulop))         # 텐서 a x b = 600
