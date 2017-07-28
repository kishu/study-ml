import tensorflow as tf

with tf.name_scope("Scope_1"):
    a = tf.add(1, 2, name="Scope_1_a")
    b = tf.multiply(a, 3, name="Scope_1_b")

with tf.name_scope("Scope_2"):
    c = tf.add(3, 4, name="Scope_2_c")
    d = tf.multiply(c, 6, name="Scope_2_d")

e = tf.add(b, d, name="output")

writer = tf.summary.FileWriter("./name_scope", graph=tf.get_default_graph())
writer.close()

