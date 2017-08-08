import tensorflow as tf

# training data
x_train = [
    [73., 80., 75],
    [93., 88., 93.],
    [89., 91., 90.],
    [96., 98., 100.],
    [73., 66., 70.]
]

y_train = [
    [80.],
    [88.],
    [91.],
    [98.],
    [66.]
]

# Model parameters
W = tf.Variable([[.3], [.3], [.3]], "weight")
b = tf.Variable([-.3], "bias")

# Model input and output
X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 1]) 

hypothesis = tf.matmul(X, W) + b

# cost/loss
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(1e-5)
train = optimizer.minimize(cost)

# run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(14000):
        _W, _b, _cost, _hypothesis, _ = sess.run([W, b, hypothesis, cost, train], {X: x_train, Y: y_train})
        print("#%s, W: %s, b: %s, hypothesis: %s, Cost: %s" % (step, _W, _b, _cost, _hypothesis))z