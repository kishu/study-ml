import tensorflow as tf

# Training Data
x_train = [
    [1, 2, 1, 1],
    [2, 1, 3, 2],
    [3, 1, 3, 4],
    [4, 1, 5, 5],
    [1, 7, 5, 5],
    [1, 2, 5, 6],
    [1, 6, 6, 6],
    [1, 7, 7, 7]]

y_train = [
    #A, B, C
    [0, 0, 1],
    [0, 0, 1], 
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0]]

# Model Input and Output
X = tf.placeholder(tf.float32, shape=(None, 4))
Y = tf.placeholder(tf.float32, shape=(None, 3))

# Model parameters
W = tf.Variable(tf.random_normal([4, 3]), name="weight")
b = tf.Variable(tf.random_normal([3]), name="bias")

# Hypothesis
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# cost/loss(Cross Entropy)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("\n# Given") 
    print("W", sess.run(W))
    print("b", sess.run(b))

    for step in range(2001):
        _W, _b, _cost, _ = sess.run([W, b, cost, train], feed_dict={X: x_train, Y: y_train})
        
        if step % 500 == 0:
            print("\n#", step) 
            print("W", _W)
            print("b", _b)
            print("cost", _cost)

    # Testing
    x_data = [
        [1, 2, 1, 1], #[0, 0, 1]
        [4, 1, 5, 5], #[0, 1, 0]
        [1, 6, 6, 6]  #[1, 0, 0]
    ]
    classification = sess.run(hypothesis, feed_dict={X: x_data})
    argmax = sess.run(tf.arg_max(classification, 1))
    print("\nClassification")
    print("result", classification)
    print("arg_max", argmax)

    writer = tf.summary.FileWriter("./01", sess.graph)
    writer.close()