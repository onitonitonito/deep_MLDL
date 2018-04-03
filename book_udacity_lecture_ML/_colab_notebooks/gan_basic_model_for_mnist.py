""" GAN_basic_model_for_MNIST.ipynb
# 비지도 학습법 / GAN (Generative Adversial Neural Network)
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

dirs = os.path.dirname(__file__).partition('k_mooc_reboot\\')
ROOT_DIR = dirs[0] + dirs[1]
DATA_DIR = ROOT_DIR + '_static\\MNIST_data\\'

mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

# Hyper- Parameter
learning_rate = 2e-4
training_epoches = 200
batch_size = 100

n_hidden = 256
n_input = 28 * 28             # 784 pix. for 1 letter
n_noise = 128

# placeholder
X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])

# Generative Side
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# Detective Side
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))

# function definition
def generator(noise_z):
    hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)
    return output


def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)
    return output


def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))


# Training set
G = generator(Z)
D_gene = discriminator(G)
D_real = discriminator(X)

cost_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
cost_G = tf.reduce_mean(tf.log(D_gene))

D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

train_D = tf.train.AdamOptimizer(
    learning_rate).minimize(-cost_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(
    learning_rate).minimize(-cost_G, var_list=G_var_list)

# Draw Graph - Neural Networ training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_batch = int(mnist.train.num_examples / batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(training_epoches):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        # calculate train(optimize), cost function as pair
        _, loss_val_D = sess.run([train_D, cost_D],
                                 feed_dict={X: batch_xs, Z: noise})

        _, loss_val_G = sess.run([train_G, cost_G],
                                 feed_dict={Z: noise})

    if epoch % 10 == 0:
        print("Epoch:%3s ___ Cost_D: %.9f" % (epoch, loss_val_D))
        print("          ___ Cost_G: %.9f\n" % (loss_val_G))

    # Check generated image
    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={Z: noise})
        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig(DATA_DIR + '{}.png'.format(
            str(epoch + 1).zfill(3)),
            bbox_inches='tight')
        plt.close(fig)

print('...optimizing finished ...')
sess.close()
