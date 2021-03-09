import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()




#################### Second dataset ####################
X_moon, y_moon = make_moons(noise=0.3, random_state=0)
X_moon = StandardScaler().fit_transform(X_moon)
X_train_moon, X_test_moon, y_train_moon, y_test_moon = \
    train_test_split(X_moon, y_moon, test_size=.4, random_state=0)


#placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#linear classification
W1 = tf.Variable(tf.random_normal([2, 50]), name="weight1")
b1 = tf.Variable(tf.random_normal([50]), name="bias1")
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([50, 1]), name="weight2")
b2 = tf.Variable(tf.random_normal([1]), name="bias2")

y_logit = tf.squeeze(tf.matmul(layer1, W2) + b2)
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
        sess.run(train, feed_dict={X: X_train_moon, Y: y_train_moon})
        if step % 1000 == 0:
            print(step, sess.run(l, feed_dict={X: X_train_moon, Y: y_train_moon}), sess.run([W1]))

    ## test and accuracy
    h, c, a = sess.run([hypothesis, Y, accuracy], feed_dict={X: X_test_moon, Y: y_test_moon})
    print("\nHypothesis: ", h, "\nTrue: ", c, "\nAccuracy: ", a)

    W, b = sess.run([W1,b1])
    print("\nW:",W,"\nb:",b)


# Plot the training points
plt.scatter(X_train_moon[:, 0], X_train_moon[:, 1], c=y_train_moon, edgecolors='k')
# Plot the testing points
plt.scatter(X_test_moon[:, 0], X_test_moon[:, 1], c=y_test_moon, alpha=0.6, edgecolors='r')


#Find and plot the wrong points
Fault = h - c
index = np.where(Fault != 0)
print(index)
print(X_test_moon[index, 0])
print(X_test_moon[index, 1])
plt.scatter(X_test_moon[index, 0], X_test_moon[index, 1],marker='x')

plt.show()