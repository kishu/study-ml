import tensorflow as tf

myvar1 = tf.Variable(4)
myvar2 = tf.Variable(5)

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

print(myvar1.eval(session=sess))