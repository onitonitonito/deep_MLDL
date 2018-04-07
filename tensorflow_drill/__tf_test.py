import numpy as np
import tensorflow as tf

x_data = np.array([[12, 12, 12],])     # x[1,3] x w[3,1] = [1]
w = np.array([[2], [4], [5]])

print(np.shape(x_data))
print(np.shape(w))


""" Draw Graph """
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

res = tf.matmul(x_data, w)
res_val = sess.run(res)
print(res_val)
