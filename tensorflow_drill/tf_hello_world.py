""" Create a constant op (operator)
This op is added as a node to the default graph
"""
import os
import sys

ROOT_DIR = "k_mooc_reboot"
CURRENT_DIR = os.path.dirname(__file__)

""" 공통 스크립트를 패치하기 위한 사전작업 """
dirs = CURRENT_DIR.partition(ROOT_DIR)
sys.path.append(dirs[0]+dirs[1])

""" 화일, DIR 관련 패치 """
import _files_dirs_run as fd
# fd.main() --- ROOT_DIR 의 화일리스트가 '장식자'로 표시된다.


import tensorflow as tf
LOG_DIR = os.path.join(fd.get_dir(), '_static', '_logdir')

def tf_hello_world(log_dir):
    """ run the op and get result
    # sess.run(hello) = <class 'type'> : bytes literals.
    # byte.decode('utf-8') = String
    """
    # 상수와 연산자를 그래프로 연결해 줘야 '텐서보드'에 표시된다.
    with tf.name_scope('Set_Constant_String'):
        hello = tf.constant("Hello Tensorflow World!!")
        say = tf.constant("World has been developed!")
        message = tf.add(hello, say)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    # Draw Graph
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print()
        print(sess.run(hello))      # return = b'Hello Tensorflow World!!'
        print(sess.run(message))

def tf_matrics_add(log_dir):
    with tf.name_scope('Setting_Constannts_Matrics'):
        a = tf.constant([[1, 2]])     # (1,2)
        b = tf.constant([[2],[3]])  # (2,1)
        c = tf.add(a, b)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    # Draw Graph
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # writer = tf.summary.FileWriter('../_logdir', sess.graph)
        c_val = sess.run(c)
        print(c_val)

if __name__ == '__main__':
    tf_hello_world(LOG_DIR)
    # tf_matrics_add(LOG_DIR)
    pass
