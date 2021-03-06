import tensorflow as tf

# Training Data
# x_data: multi variable
# y_data: binary classification 0: fail 1: pass

# X1이 1, X2가 2일때 Y 즉 결과는 0이며 fail
# X1이 3, X2가 1일때 Y 즉 결과는 1이며 true
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

# placeholders for a tensor that will be always fed
# X는 2의 크기를 같는 노드를 None 즉 N개까지 가질 수 있다.
X = tf.placeholder(tf.float32, shape=[None, 2])
# Y는 1의 크기를 같는 노드를 None 즉 N개까지 가질 수 있다.
Y = tf.placeholder(tf.float32, shape=[None, 1])

# [2, 1]: x feature 2개를 받아 1개의 W를 만든다.
W = tf.Variable(tf.random_normal([2, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b))

'''
          1
H(X) = —————————
       1 + e^WtX
'''

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost
'''
cost(W) = - 1/M ∑ylog(H(x)) + (1-y)(log(1- H(x))
'''

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# minimize
'''
            ∂
W := W - α ———— cost(W)
            ∂W
'''
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy(정확하게) computation
# if hypothesis > 0.5 return 1.0 that float32 casting of "true"
# if hypothesis <= 0.5 return 0.0 that float32 casting of "false"
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

# 예측값(predicted)와 실제값(Y)가 맞는지 평균을 구한다.
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


# Train the model
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

