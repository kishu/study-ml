import tensorflow as tf

# Pass in a starting value of three for the variable
my_var = tf.Variable(4, name="my_variable")
add = tf.add(5, my_var)
mul = tf.multiply(8, my_var)

sess = tf.Session()
sess.run(my_var.initializer)

print(my_var.eval(session=sess))
print(add.eval(session=sess))
print(mul.eval(session=sess))

