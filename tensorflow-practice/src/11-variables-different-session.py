import tensorflow as tf

import tensorflow as tf



my_var = tf.Variable(0)
init = tf.global_variables_initializer()

# start sessions
sess1 = tf.Session()
sess2 = tf.Session()

# initialize variable in sess1, and increment value of my_var in that session
sess1.run(init)
sess1.run(my_var.assign_add(1))    # 1
print(my_var.eval(session=sess1))

# do the same with sess2, but use a different increment value
sess2.run(init)
sess2.run(my_var.assign_add(2))    # 2
print(my_var.eval(session=sess2))

# can increment the variable values in each session independently
sess1.run(my_var.assign_add(5))    # 6
sess2.run(my_var.assign_add(2))    # 4
print(my_var.eval(session=sess1))
print(my_var.eval(session=sess2))
