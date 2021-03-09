import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


#################### Third dataset ####################
X_circle, y_moon_circle = make_circles(noise=0.2, factor=0.5, random_state=0)
X_circle = StandardScaler().fit_transform(X_circle)
X_train_circle, X_test_circle, y_train_circle, y_test_circle = \
    train_test_split(X_circle, y_moon_circle, test_size=.4, random_state=0)

#placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#linear classification
W1 = tf.Variable(tf.random_normal([2, 1]), name="weight1")
b1 = tf.Variable(tf.random_normal([1]), name="bias1")

y_logit = tf.squeeze(tf.matmul(X, W1) + b1)
y_one_prob = tf.sigmoid(y_logit)
hypothesis = tf.round(y_one_prob)

# cost function / minimize cost
entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=Y)
l = tf.reduce_sum(entropy)
train = tf.train.AdamOptimizer(.1).minimize(l)

# predicate / accuracy
#predicated = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(hypothesis, Y), dtype=tf.float32))

# session run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(5000):
        sess.run(train, feed_dict={X: X_train_circle, Y: y_train_circle})
        if step % 1000 == 0:
            print(step, sess.run(l, feed_dict={X: X_train_circle, Y: y_train_circle}), sess.run([W1]))

    ## test and accuracy
    h, c, a = sess.run([hypothesis, Y, accuracy], feed_dict={X: X_test_circle, Y: y_test_circle})
    print("\nHypothesis: ", h, "\nTrue: ", c, "\nAccuracy: ", a)

    W, b = sess.run([W1,b1])
    print("\nW:",W,"\nb:",b)

# Plot the training points
plt.scatter(X_train_circle[:, 0], X_train_circle[:, 1], c=y_train_circle, edgecolors='k')
# Plot the testing points
plt.scatter(X_test_circle[:, 0], X_test_circle[:, 1], c=y_test_circle, alpha=0.6, edgecolors='r')

#Find and plot the wrong points
Fault = h - c
index = np.where(Fault != 0)
print(index)
print(X_test_circle[index, 0])
print(X_test_circle[index, 1])
plt.scatter(X_test_circle[index, 0], X_test_circle[index, 1],marker='x')

plt.show()
