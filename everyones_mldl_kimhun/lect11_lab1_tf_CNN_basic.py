""" ERROR

"""
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='mnist_img')
X_img = tf.reshape(tensor=X, shape=[-1,28,28,1])        # image reshape
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='hypothesis')

""" Hyper Parameter """
nb_classes = 64             # number of choices 64
training_epoches = 15
learning_rate = 1e-2
keep_prob = tf.placeholder(tf.float32)


""" LAYER-01, HIDDEN
# Tensor("conv2d: 0",    shape=(?,28,28,32), dtype=float32)
# Tensor("Relu: 0",      shape=(?,28,28,32), dtype=float32)
# Tensor("Maxpool: 0",   shape=(?,14,14,32), dtype=float32)
# Tensor("Dropout/mul:0",shape=(?,14,14,32), dtype=float32)
"""
W1 = tf.Variable(tf.random_normal(
    shape=[3, 3, 1, 32], stddev=1e-1, dtype=tf.float32,
    mean=0.0, name='W1-weight'))

""" NN function : (1) conv2d --> (2) relu --> (3) max_pool --> (4) dropout
# Conv --> (?, 28, 28, 32)
# Pool --> (?, 14, 14, 32)
"""
L1 = tf.nn.conv2d(
    input=X_img, filter=W1, strides=[1, 1, 1, 1], padding='SAME',
    name='L1-convolution-2d')
L1 = tf.nn.relu(features=L1, name='L1-relu')
L1 = tf.nn.max_pool(
    value=L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
    name='L1-max_pool')
L1 =  tf.nn.dropout(x=L1, keep_prob=keep_prob, name='L1-dropout')


""" LAYER-02, HIDDEN
# Tensor("conv2d: 0",    shape=(?,14,14,64), dtype=float32)
# Tensor("Relu: 0",      shape=(?,14,14,64), dtype=float32)
# Tensor("Maxpool: 0",   shape=(?,14,14,42), dtype=float32)
# Tensor("Dropout/mul:0",shape=(?,14,14,32), dtype=float32)
"""
W2 = tf.Variable(tf.random_normal(
    shape=[3, 3, 32, 64], stddev=1e-1, dtype=tf.float32,
    mean=0.0, name='w2-weight'))

""" NN function : (1) conv2d --> (2) relu --> (3) max_pool --> (4) dropout
# Conv --> (?, 14, 14, 64)
# Pool --> (?,  7,  7, 64)
"""
L2 = tf.nn.conv2d(
    input=L1, filter=W2, strides=[1, 1, 1, 1], padding='SAME',
    name='L2-convolution-2d')
L2 = tf.nn.relu(features=L2, name='L2-relu')
L2 = tf.nn.max_pool(
    value=L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
    name='L2-max_pool')
L2 =  tf.nn.dropout(x=L2, keep_prob=keep_prob, name='L2-dropout')


""" LAYER-03, HIDDEN : shape = (?,7,7,64)
# Tensor("conv2d: 0",    shape=(?,7,7,128), dtype=float32)
# Tensor("Relu: 0",      shape=(?,7,7,128), dtype=float32)
# Tensor("Maxpool: 0",   shape=(?,4,4,128), dtype=float32)
# Tensor("Dropout/mul:0",shape=(?,4,4,128), dtype=float32)
# ------------
# Tensor("Reshape_1: 0", shape=(?, 2048), dtype=float32)
"""
W3 = tf.Variable(tf.random_normal(
    shape=[3, 3, 32, 64], stddev=1e-1, dtype=tf.float32,
    mean=0.0, name='w2-weight'))

""" NN function : (1) conv2d --> (2) relu --> (3) max_pool --> (4) dropout
# Conv --> (?, 7, 7, 128)
# Pool --> (?, 4, 4, 128)
# Reshape->(?, 4 * 4 * 128) = flatten for FC ('Fully connected')
"""
L3 = tf.nn.conv2d(
    input=L2, filter=W3, strides=[1, 1, 1, 1], padding='SAME',
    name='L3-convolution-2d')
L3 = tf.nn.relu(features=L3, name='L3-relu')
L3 = tf.nn.max_pool(
    value=L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
    name='L3-max_pool')
L3 =  tf.nn.dropout(x=L3, keep_prob=keep_prob, name='L3-dropout')

""" final reshap down """
L3 = tf.reshape(tensor=L3, shape=[-1, 128 * 4 * 4])  # 128*4*4=2048


""" L4 : FC(Fully-Connected) LAYER = 4x4x128 inputs --> 625 outputs
# Tensor("Relu_3:0",        shape=(?, 625), dtype=float32)
# Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
"""
W4 = tf.get_variable(
    name="W4", shape=[128 * 4 * 4, 625],
    initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal(shape=[625]))
L4 = tf.nn.relu(features=tf.matmul(L3, W4) + b4, name='L4-relu')
L4 = tf.nn.dropout(x=L4, keep_prob=keep_prob, name='L4-dropout')


""" L5 final FC-LAYER : 625 inputs --> 10 outputs
"""
W5 = tf.get_variable(
    name="W5", shape=[625, 10],
    initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal(shape=[10]))

""" Tensor("add_1:0", shape=(?, 10), dtype=float32) """
hypothesis = tf.matmul(L4, W5) + b5


""" Test model & check accuracy """
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(
    x=correct_prediction, DstT=tf.float32, name='accuracy'))




accu_val = se
print("".format())


"""
# cost optimaize funtion
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=hypothesis,
    labels='with_logits'))

# train
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.glorot_normal_initializer())

    for epoch in range(training_epoches):
        average_cost = 0
        total_batch = int(mnist.train.num_examples / total_batch)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X:batch_xs, Y:batch_ys}

            cost_val, _ = sess.run([cost, train], feed_dict=feed_dict)
            average_cost += cost_val / total_batch

        print("Epoch: {:4d}{:.9f}".format(
            epoch+1,
            average_cost))

    print('... learnig is finished! ...')

with tf.InteractiveSession() as sess:
    sess.run(tf.global_variables_initializer())

    print("image.shape")
    img = mnist.train.images[0].reshape(1, 28, 28, 1)
    # img = img.shape(-1, 28, 28, 1)
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=1e-1))
    conv2d = tf.nn.conv2d (img, W1, strides=[1,2,2,1], padding='SAME')
    print(conv2d)

    conv2d_img = conv2d.eval()
    conv2d_img = np.swapaxes(conv2d_img, 0, 3)

    for i, one_img in enumerate(conv2d_img):
        plt.subplot(1, 5, i+1), plt.imshow(one_img.reshape(14,14), cmap='gray')
"""
