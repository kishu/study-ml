import tensorflow as tf

g = tf.Graph()

with g.as_default():
    c = tf.constant(30)

assert c.graph is g
indefaultgraph = tf.add(1, 2, name="indefaultgraph")

with g.as_default():
    ingraphg = tf.multiply(2, 3, name="ingraphg")

alsoindefraph = tf.subtract(5, 1, name="alsoindefgraph")

sess = tf.Session()
