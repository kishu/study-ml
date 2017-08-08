import tensorflow as tf
import numpy as np

# Trainning data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Create trainable value
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# Hypothesis: XW + b
hypothesis = tf.add(tf.multiply(W, x_train), b)

# Cost function
cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, y_train)))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)

# Launch the graph in a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1000):
    sess.run(optimizer.minimize(cost))
    print("#", step, "cost =", sess.run(cost), "W =", sess.run(W), "b =", sess.run(b))





# import tensorflow as tf

# # 1
# # Build graph using TF operation

# #H(x) = Wx + b
# x_train = [1, 2, 3]
# y_train = [1, 2, 3]

# # Create trainable value
# W = tf.Variable(tf.random_normal([1]), name="weight")
# b = tf.Variable(tf.random_normal([1]), name="bias")

# # Hypothesis XW + b
# hypothesis = x_train * W + b

# # cost function
# cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# # t = [1., 2., 3., 4.]
# # tf.reduce_mean(t) # 2.5

# # Minimize
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
# train = optimizer.minimize(cost)


# # 2 - 3
# # Run/update graph and get result

# # Launch the graph in a session
# sess = tf.Session()

# # Initialize global variables in the graph
# # use W, b of tf.variable
# sess.run(tf.global_variables_initializer())

# # Fit the line
# for step in range(2001):
#     sess.run(train)
#     print(step, sess.run(cost), sess.run(W), sess.run(b))
