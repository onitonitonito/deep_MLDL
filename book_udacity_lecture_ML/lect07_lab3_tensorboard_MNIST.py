"""
Graph and Loss visualization using Tensorboard.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
# from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 1e-1
nb_classes = 10
training_epochs = 15
batch_size = 100
display_epoch = 1
logdir = './_logdir/'

""" MNIST image of shape = 28 * 28 = 784 pixel
# X:[n, 784] * W:[784, 10] + b[10] = Y:[n, 10]
# X:n = 트레이닝 데이타의 겟수 / Y:n = 질문의 갯수가 될 수 있다.
# 784개의 데이터를 읽어야, 0~9 중 택1을 할수 있다 : softmax().argmax()
"""
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

# Set model weights
W = tf.Variable(tf.zeros([784, 10]), name='Weights')
b = tf.Variable(tf.zeros([10]), name='Bias')

# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model-softmax'):                # Model
    hypothesis = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

with tf.name_scope('Loss-cost'):        # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))

with tf.name_scope('Optimizer'):        # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

with tf.name_scope('Accuracy'):         # Accuracy
    is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Create a summary to monitor cost tensor
# Create a summary to monitor accuracy tensor
# Merge all summaries into a single op
tf.summary.scalar("minimize cost function", cost)
tf.summary.scalar("accuracy of Prediction", accuracy)
merged_summary_op = tf.summary.merge_all()

# Start training & Draw graph session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        average_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # total_batch = 550, num_examples = 55k

        # train(cost minimize) = 100 x 550 repeats = 1 Epoch = whole data set
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, cost_val, summary = sess.run(
                fetches=[train, cost, merged_summary_op],
                feed_dict={x: batch_xs, y: batch_ys})
            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            average_cost += cost_val / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_epoch == 0:
            print("Epoch:{:3d} ___ Cost:{:.9f}".format(
                epoch + 1,
                average_cost))
    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    print("Accuracy:", accuracy.eval(
        session=sess,
        feed_dict={x: mnist.test.images, y: mnist.test.labels}))

    print("Run the command line:\n" +\
          "--> tensorboard --logdir=./book_udacity_lecture_ML/_logdir/\n" +\
          "Then open http://your-site:6006/ into your web browser")
