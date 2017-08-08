import tensorflow as tf

a = tf.constant(6, name="constant_a")
b = tf.constant(3, name="constant_b")
c = tf.constant(10, name="constant_c")
d = tf.constant(5, name="constant_d")

mul = tf.multiply(a, b, name="mul")
