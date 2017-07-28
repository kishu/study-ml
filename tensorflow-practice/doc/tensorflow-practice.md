TensorFlow
===

---

# installation

## Requirements
* [python3](https://www.python.org/)
* [homebrew](https://brew.sh/)
* update xcode if installed 


``` sh
pip3 install numpy six wheel
brew install coreutils
```
---

# installation

``` sh
cd /your/workspace
git clone https://github.com/tensorflow/tensorflow
checkout <release_tag>
./configuration
```

```sh
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip3 install /tmp/tensorflow_pkg/<whl>
```


---

# Computation Graph Basics

![80% computation-graph-basics](img/computation-graph-basics.png)


``` python
a = tf.constant(2, name="input_a")
b = tf.constant(3, name="input_b")

c = tf.add(a, b, name="add_c")
```

---

# Dependency in Computation Graphs

![80% dependency-in-computation-graphs](img/dependency-in-computation-graphs.png)

``` python
a = tf.constant(4, name="input_a")
b = tf.constant(3, name="input_b")

c = tf.add(a, b, name="add_c")
d = tf.multiply(a, b, name="mul_d")
e = tf.multiply(c, d, name="mul_e")
```

---

Tensors and Graphs
===

---

# Using Tensor

![80% using-tensor](img/using-tensor.png)

``` python
a = tf.constant([4, 3], name="constant_a")

# Computes the sum of elements across dimensions of a tensor.
b = tf.reduce_sum(a, name="sum_b")

# Computes the product of elements across dimensions of a tensor.
c = tf.reduce_prod(a, name="mul_c")

d = tf.multiply(b, c, name="mul_d")
```

---

# Tensor from Python Native Types

Treated as 0-D Tensor, or ***"scalar"***
`t_0 = 50` 

Treated as 1-D Tensor, or ***"vector"***
`t_1 = [1, 3, 3]`

treated as 2-D Tensor, or ***"matrix"***
``` python
t_w = [
  [True, True, False],
  [False, Falsse, True],
  [False, True, False]
]
```

---

# Tensor from NumPy Arrays

0-D Tensor with 32-bit integer data type
`t_0 = np.array(50, dtype=np.int32)`

1-D Tensor with byte string data type
Node: Don't explicitly specify dtype when using strings in NumPy
`t_1 = np.array([b"apple", b"peach", b"grape"])`

2-D Tensor with boolean data type
``` python
t_2 = np.array([
  [True, False, False],
  [False, False, True],
  [False, True, False]
], dtype=np.bool);
```
---

# Tensor Shape

Shape that describes a vector of length
`[1, 2, 3]` has shape `[3]`

Shape that describes matrix

```
[[1,2],
 [3,4],
 [5,6]]
 ```
 has shape `[3, 2]`
 
---
 
# Tensorflow Operations
 
## Add x and y, elementwise

`x + y`
`tf.add()` 
 
## Subtract y from x, elementwise

`x - y`
`tf.subtract()`
 
---

# Tensorflow Operations

## Multiply x and y, elementwise

`x * y`
`tf.multiply()` 

## Divides x / y elementwise 

`x / y`
`tf.div()`

Perform element-wise integer division when givien an integer type tensor, and floating point("true") division on floating point tensors.

---

Session and Run
===

---

# TensorFlow Sessions

``` python
# Build a graph
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

# Launch the graph in a session
sess = tf.Session()

# Evaluate the tensor 'c'
output = sess.run(c)
print(output)
```

---

# Close Sessions

Using the 'close()' method.

``` python
sess = tf.Session()
sess.run(...)
sess.close()
```

Using the context manager

``` python
with tf.Session() as sess:
  sess.run(...)
```

---

# Tensor.eval

``` python
a = tf.constant(3)
sess = tf.Session()

# Use the Session as a default inside of 'with' block
with sess.as_default():
  a.eval()

# Have to close Session manually
sess.close()
```

----

# Session Optional Arguments

``` python
tf.Session.__init__(target="", graph=None, config=None)
```

* target: (Optional) The exection engine to connect to. Defaults to using an in-process engine.
* graph: (Optional) The Graph to be launched(described above).
* config:(Optional) A ConfigProto protocol buffer with configuration options for the session.

---

# Running a Session

``` python
tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None)
```

``` python
a = tf.constant([10, 20])
b = tf.constant([1.0, 2.0])

sess = tf.Session()

# 'fetches' can be a singleton
# 'v' is [10, 20]
v = sess.run(a)

# 'fetches' can be a list
# 'v' is [array([10, 20], dtype=int32), array([ 1.,  2.], dtype=float32)]
v = sess.run([a, b])
```
---

# Example for the feed_dictionary

``` python
# Create Operations, Tensor, etc(using default graph)
a = tf.add(2, 3)
b = tf.multiply(a, 4)

# Start up a 'Session' using the default graph
sess = tf.Session()

# Define a dictionary that says to replace the value of 'a' with 15
replace_dict = {a: 15}

# Run the session, passing in 'replace_dict' as the value to 'feed_dict'
sess.run(b, feed_dict=replace_dict) # return 60
```

---

# Inputs with Placeholder

``` python
tf.placeholder(dtype, shape=None, name=None)
```

``` python
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
	print(sess.run(y)) # ERROR: will fail because x was not fed.
    
    rand_array = np.random.rand(1024, 1024)
    print(sess.run(y, feed_dict={x: rand_array}))
```
---

# Another Example for Placeholder

```
import tensorflow as tf
import numpy as np

a = tf.placeholder(tf.int32, shape=[2], name="input")
b = tf.reduce_prod(a, name="prod_b") # 2
c = tf.reduce_sum(a, name="sum_c")   # 3
d = tf.add(b, c, name="add_d")       # 5

sess = tf.Session()
input_dict = {a: np.array([1, 2], dtype=np.int32)}

# Fetch the value og 'd', feeding the values of 'input_vactor' into 'a'
sess.run(d, feed_dict=input_dict)    # 5
```

---

Variables
===

---

# Variables

Tensor and Operation objects are immutable

``` python
# Pass in a starting value of three for the variable
my_var = tf.Variable(4, name="my_variable")
add = tf.add(5, my_var)
mul = tf.multiply(8, my_var)

sess = tf.Session()
sess.run(my_var.initializer)

print(my_var.eval(session=sess))    # 4
print(add.eval(session=sess))       # 9
print(mul.eval(session=sess))       # 32
```

---

# Variables

``` python
# 3x2 matrix of zeros
zeros = tf.zeros([3, 2])

# vector of length 4 of ones
ones = tf.ones([4])

# 4x4 Tensor of random uniform values between 0 and 10
uniform = tf.random_uniform([4, 4], manval=0, maxval=10)

# 4x3x2 Tensor of normally distributed numbers; mean(평균) 0 deviation(편차) 3
normal = tf.random_normal([4, 3, 2], mean=0.0, stddev=3.0)
```

``` python
var = tf.Variable(zeros)
sess.run(var.initializer)

var.eval(session=sess)
```

---

# Variable Initialization

``` python
# Launch The Graph in a Session
with tf.Session() as sess:

  # Run The Variable Initializer
  sess.run(w.initializer)
  # ... you now can run ops that use the value of 'w'
```

---

# Variable Initialization

``` python
myvar1 = tf.Variable(4)
myvar2 = tf.Variable(5)

# Add an Op to initialize global variables
init_op = tf.global_variables_initializer()

# Launch the graph in a session
sess = tf.Session()

# Run the Op that initializes global variable
sess.run(init_op)

# ... you can now run any Op that uses variable values...
print(myvar1.eval(session=sess))
```

---

# Changing Variables

``` python
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
```

---

# Incrementiong and Decrementiong

``` python
# Increment by 1
sess.run(my_var.assign_add(1))

# Decrement by 1
sess.run(my_var.assign_sub(1))
```

---

# Variables in Different Sessions

``` python
my_var = tf.Variable(0)
init = tf.global_variables_initializer()

# start sessions
sess1 = tf.Session()
sess2 = tf.Session()

# initialize variable in sess1, and increment value of my_var in that session
sess1.run(init)
sess1.run(my_var.assign_add(1))    # 1

# do the same with sess2, but use a different increment value
sess2.run(init)
sess2.run(my_var.assign_add(2))    # 2

# can increment the variable values in each session independently
sess1.run(my_var.assign_add(5))    # 6
sess2.run(my_var.assign_add(2))    # 4
```

---

# Trainable Variables

```
non_trainable_var = tf.Variable(0, trainable=False)
```

---

Organize You Graphs
===

---

# Name Scopes for Graphs

Simply add your Operations in a with `tf.name_scope(<name>)` block

``` python
with tf.name_scope("Scope1"):
  a = tf.add(1, 2, name="Scope1_add")
  b = tf.multiply(a, 3, name="Scope1_mul")

with tf.name_scope("Scope2"):
  c = tf.add(4, 5, name="Scope2_add")
  d = tf.multiply(c, 6, name="Scope2_mul")

e = tf.add(b, d, name="output")
```

---

# Nest Name Scopes

``` python
with tf.name_scope("transformation"):
  with tf.name_scope("A)
    a_add = tf.add(1, 2)
    b_mul = tf.multiply(a_add, 3)
  with tf.name_scope("B")
    b_add = tf.add(4, 5)
    b_mul = tf.mul(b_add, 6)
```