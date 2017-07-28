import tensorflow as tf

# Create Operations, Tensor, etc(using default graph)
a = tf.add(2, 3)
b = tf.multiply(a, 4)

# Start up a 'Session' using the default graph
sess = tf.Session()

# Define a dictionary that says to replace the value of 'a' with 15
replace_dict = {a: 15}

# Run the session, passing in 'replace_dict' as the value to 'feed_dict'
o = sess.run(b, feed_dict=replace_dict) # return 60

print(o)