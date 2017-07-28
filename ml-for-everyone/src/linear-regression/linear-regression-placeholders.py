import tensorflow as tf

# 1
# Build graph using TF operation

#H(x) = Wx + b
# x_train = [1, 2, 3]
# y_train = [1, 2, 3]

# Using placeholders
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Create trainable value
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Hypothesis XW + b
hypothesis = X * W + b

# cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# t = [1., 2., 3., 4.]
# tf.reduce_mean(t) # 2.5

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

# 2 - 3
# Run/update graph and get result

# Launch the graph in a session
sess = tf.Session()

# Initialize global variables in the graph
# use W, b of tf.variable
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
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
