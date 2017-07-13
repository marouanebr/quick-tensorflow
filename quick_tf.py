# A quick tutorial for TensorFlow.
# Source: http://cv-tricks.com/artificial-intelligence/deep-learning/deep-learning-frameworks/tensorflow-tutorial/

# Part-1: Basics of TensorFlow
import tensorflow as tf
import numpy as np

# (i) Graph in TensorFlow
# Everything that happens in the code, resides on a default graph provided by TensorFlow
graph = tf.get_default_graph()

# show operations in a graph
for op in graph.get_operations():
    print(op.name)

# (ii) TensorFlow Session
# Graph only defines the computations or builds the blueprint.
# However, there are no variables, no values unless we run the graph within a session
# to run a session
# with tf.Session() as sess:
#     sess.run(f)
# or
# sess=tf.Session()
# pass
# pass
# sess.close()

# (iii) Tensors in TensorFlow
# a) Constants
a = tf.constant(1.0)
print('outside of a session, a is ', a)
with tf.Session() as sess:
    print('inside of a session, a is ', sess.run(a))

# b) Variables
# they need to be initialized by an init op.
b = tf.Variable(2.0, name="test_var")
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(b))

for op in graph.get_operations():
    print(op.name)

# c) Placeholders
# What is fed to Placeholder is called feed_dict. Feed_dict are key value pairs for holding data
c = tf.placeholder("float")
d = tf.placeholder("float")
y = tf.multiply(c, d)
feed_dict = {c: 2, d: 3}
with tf.Session() as sess:
    print(sess.run(y, feed_dict))

# Part-2: Tensorflow tutorial with simple example

# Create a random normal distribution:
w = tf.Variable(tf.random_normal([784, 10], stddev=0.01))

# Reduce_mean
b = tf.Variable([10,20,30,40,50,60],name='t')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.reduce_mean(b))

# ArgMax
a=[ [0.1, 0.2, 0.3 ], [20, 2, 3]  ]
b = tf.Variable(a,name='b')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.argmax(b,1))

trainX = np.linspace(-1, 1, 101)
trainY = 3 * trainX + np.random.randn(*trainX.shape) * 0.33

X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(0.0, name="weights")
y_model = tf.multiply(X, w)
cost = (tf.pow(Y-y_model, 2))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        for (x, y) in zip(trainX, trainY):
            sess.run(train_op, feed_dict={X: x, Y: y})
    print(sess.run(w))