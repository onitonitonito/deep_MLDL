import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)                 # for reproducibility

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
# mnist = input_data.read_data_sets("", one_hot=True)

# Hyper variable Setting
nb_classes = 10
learning_rate = 1e-2
training_epoches = 20
batch_size = 100

n_hidden = 256
n_input = 28*28                         # 784 pix. for 1 letter

# placeholder
X = tf.placeholder(tf.float32, [None, n_input])

# encoder
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))
encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))

# decoder
W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode), b_decode))

cost = tf.reduce_mean(tf.pow(X-decoder, 2))
train = tf.train.RMSPropOptimizer(learning_rate)
optimizer = train.minimize(cost)

# run graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

num_examples = mnist.train.num_examples
print(num_examples, batch_size, int(num_examples / batch_size))
total_batch = int(num_examples / batch_size)        # 550k / 100 = 550

for epoch in range(training_epoches):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X:batch_xs})
        total_cost += cost_val
    print("Epoch: %s __ Avg.Cost: %s"% (epoch + 1, total_cost/total_batch))
print("... Learning Finished! ...   TOTAL COST: %s"% total_cost)

# test set 10 samples
sample_size = 10
samples = sess.run(decoder, feed_dict={X:mnist.test.images[:sample_size]})

fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))
for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()

    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
    ax[1][i].imshow(np.reshape(samples[i], (28,28)))
plt.show()
sess.close()
