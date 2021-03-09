import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()




#################### First dataset ####################
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)

X = StandardScaler().fit_transform(X)
X_train_linear, X_test_linear, y_train_linear, y_test_linear = \
    train_test_split(X, y, test_size=.4, random_state=0)


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
        sess.run(train, feed_dict={X: X_train_linear, Y: y_train_linear})
        if step % 1000 == 0:
            print(step, sess.run(l, feed_dict={X: X_train_linear, Y: y_train_linear}), sess.run([W1]))

    ## test and accuracy
    h, c, a = sess.run([hypothesis, Y, accuracy], feed_dict={X: X_test_linear, Y: y_test_linear})
    print("\nHypothesis: ", h, "\nTrue: ", c, "\nAccuracy: ", a)

    W, b = sess.run([W1,b1])
    print("\nW:",W,"\nb:",b)


# Plot the training points
plt.scatter(X_train_linear[:, 0], X_train_linear[:, 1], c=y_train_linear, edgecolors='k')
# Plot the testing points
plt.scatter(X_test_linear[:, 0], X_test_linear[:, 1], c=y_test_linear, alpha=0.6, edgecolors='r')


#Find and plot the wrong points
Fault = h - c
index = np.where(Fault != 0)
print(index)
print(X_test_linear[index, 0])
print(X_test_linear[index, 1])
plt.scatter(X_test_linear[index, 0], X_test_linear[index, 1],marker='x')

plt.show()