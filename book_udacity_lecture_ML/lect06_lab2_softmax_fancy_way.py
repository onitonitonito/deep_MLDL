""" Animal_Classification w/ Softmax_cross_entropy_with_logits()
# 소프트맥스-크로스엔토로피-디짓즈, 팬시웨이를 사용한다
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

CSV_DIR = os.path.join(os.path.dirname(__file__), "_csv_hunkim/")
tf.set_random_seed(777)     # for reproducibility (재현성)

# Lab - code / X:[n x 16] * W:[16 x 7] = Y:[n x 7] , nb_classes = 7 = [1,0,0,0,0,0,0]
# Prediction animal type based on various features
xy = np.loadtxt(CSV_DIR+'data04_zoo_classify.csv', delimiter=',', dtype=np.float32)
x_data = xy[ :, 0:-1]
y_data = xy[ :, [-1] ]

""" data shape.chck!! """
print("x_data= ", x_data.shape, "\ny_data.shape= ", y_data.shape, "\n")

nb_classes = 7            # 0~6 중, 택1

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])           # 0~6 중, 택1
Y_one_hot = tf.one_hot(Y, nb_classes)        # 한 차원이 늘어남
print(" (1) one_hot (original) = ",Y_one_hot)

Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print(" (2) one_hot (reshaped) = ",Y_one_hot)

print()
# nb_classes (답변의 종류) = 7 .. 동물의 종류
W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.sotfmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# cost() = cross-entropy() w/ logits
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(cost)

# Lab - code : # Prediction's Accuracy check
# prediction's Accuracy check
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch the Graph = Learning
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./_logdir', sess.graph)

    loss_array, accuracy_array = [], []
    print()
    for step in range(2001):
        sess.run(optimizer, feed_dict = {X:x_data, Y:y_data})
        if step%50 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict = {X:x_data, Y:y_data})
            loss_array.append(loss)
            accuracy_array.append(acc * 5)
            print("{:4}. __ loss= {:.3f} / accuracy= {:6.2f}%".format(
                step,
                loss,
                acc * 100))
    print()
    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict = {X:x_data, Y:y_data})

    # y_data: (N,1) = flatten => (N,) matches pred.shape
    # for p, y in zip(pred, y_data.flatten()):
    #     print(p == int(y), p, int(y))  # p=prediction, y=answer(fact)

y1, y2 = loss_array, accuracy_array
plt.plot(y1, lw=1, label='loss')
plt.plot(y2, lw=2, label='accuracy')
plt.legend()
plt.grid()
plt.show()
