import tensorflow as tf

my_var = tf.Variable(0)

# Create an operation that multiplies the variable by 2 each time it is run 
my_var_plus_two = my_var.assign(my_var + 2)

# Initialization
init = tf.global_variables_initializer()

# Start a Session
sess = tf.Session()

# Initialize variable
sess.run(init)

print(sess.run(my_var))    # 0

# add variable by two and return it
print(sess.run(my_var_plus_two))  # 2

# my var changed!!!
print(sess.run(my_var))    # 2