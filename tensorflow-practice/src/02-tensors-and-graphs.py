import tensorflow as tf
import numpy as np

'''
# Treated as 0-D Tensor, or "scalar"
t_0 = 50

# Treated as 1-D Tensor, or "vector"
t_1 = [1 ,2,3]

# Treated as 2-D Tensor, or "matrix"
t_2 = [
    [True, True, False],
    [False, False, True],
    [False, True, False]
]
'''

'''
Tensor from NumPy arrays

# 0-D Tensor with 32-bit integer data type
t_0 = np.array(50, dtype=np.int32)

# 1-D Tensor with byte string data type
# Note: Don't explicitly specify dtype when using strings in NumPy
t1 = np.array([b"apple", b"peach", b"grape"])

# 2-D Tensor with boolean data type
t2 = np.array([
    [true, False, False],
    [False, False, True],
    [False, True, False]])

'''



a = tf.constant([4, 3], name="constant_a")
b = tf.reduce_sum(a, name="sum_b")
c = tf.reduce_prod(a, name="mul_c")
d = tf.multiply(b, c, name="mul_d")

''' using native type
nt = [4, 3]
a = tf.constant(nt, name="constant_a")
'''

''' using numpy
npt = ap.array([4,3], dtype=np.int32)
a = tf.constant(npt, name="constant_a")

sess = tf.Session()
writer = tf.summary.FileWriter("./graph-02", sess.graph)
output = sess.run(d)

print(output)

'''
$ tensorboard --logdir="graph-02"
'''
