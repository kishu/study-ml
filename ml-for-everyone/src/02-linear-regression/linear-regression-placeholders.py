import tensorflow as tf
import numpy as np

# Using placeholders
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Create trainable value
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# Hypothesis XW + b
pred = tf.add(tf.multiply(W, X), b)

# cost function
cost = tf.reduce_mean(tf.square(tf.subtract(pred, Y)))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)

# Launch the graph in a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(optimizer.minimize(cost),)
    
    
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
        feed_dict = {
            X: [1, 2, 3, 4, 5],
            Y: [2.1, 3.1, 4.1, 5.1, 6.1]
        })
    print(step, cost_val, W_val, b_val)


# Testing our model
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))
