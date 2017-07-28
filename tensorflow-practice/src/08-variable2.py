import tensorflow as tf

# 3x2 matrix of zeros
zeros = tf.zeros([3, 2])

# vector of length 4 of ones
ones = tf.ones([4])

# 4x4 Tensor of random uniform values between 0 and 10
uniform = tf.random_uniform([4, 4], minval=0, maxval=10)

# 4x3x2 Tensor of normally distributed numbers; mean(평균) 0 deviation(편차) 3
normal = tf.random_normal([4, 3, 2], mean=0.0, stddev=3.0)

sess = tf.Session()
o = sess.run([zeros, ones, uniform, normal])

print(o)

var = tf.Variable(zeros)
sess.run(var.initializer)

o = var.eval(session=sess)

print(o)
