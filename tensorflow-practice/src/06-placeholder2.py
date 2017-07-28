import tensorflow as tf
import numpy as np

a = tf.placeholder(tf.int32, shape=[2], name="input")
b = tf.reduce_prod(a, name="prod_b") # 2
c = tf.reduce_sum(a, name="sum_c")   # 3
d = tf.add(b, c, name="add_d")       # 5

sess = tf.Session()
input_dict = {a: np.array([1, 2], dtype=np.int32)}

# Fetch the value og 'd', feeding the values of 'input_vactor' into 'a'
o = sess.run(d, feed_dict=input_dict)
print(o)