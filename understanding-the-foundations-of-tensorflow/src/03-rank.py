import tensorflow as tf

sess = tf.Session()

zeroD = tf.constant(5)
print(sess.run(tf.rank(zeroD)))