import tensorflow as tf

A = tf.constant([4], tf.int32, name="A")
B = tf.constant([5], tf.int32, name="B")
C = tf.constant([6], tf.int32, name="C")

x = tf.placeholder(tf.int32, name="x")

# y = Ax^2 + Bx + C
with tf.name_scope("Equation_1"):
    Ax2 = tf.multiply(A, tf.pow(x, 2), name="Ax2")
    Bx = tf.multiply(B, x, name="Bx")
    y1 = tf.add_n([Ax2, Bx, C], name="calc_1")

# y = Ax^2 + Bx
with tf.name_scope("Equation_2"):
    Ax2 = tf.multiply(A, tf.pow(x, 2), name="Ax2")
    Bx = tf.multiply(B, x, name="Bx")
    y2 = tf.add_n([Ax2, Bx], name="calc_1")

with tf.name_scope("Final_Sum"):
    y = y1 + y2

with tf.Session() as sess:
    print(sess.run(y, feed_dict={x: [10]}))
    
    writer = tf.summary.FileWriter("./09", sess.graph)
    writer.close()
