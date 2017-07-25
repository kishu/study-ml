import tensorflow as tf

# add as a node to default graph
hello = tf.constant("hello tensorflow")

# start tf session
sess = tf.Session()

# run
print(sess.run(hello))
