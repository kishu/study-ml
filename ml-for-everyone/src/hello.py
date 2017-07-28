'''
TensorFlow is an open source software library for
numerical computation using data flow graph

TensorFlow는 데이터 플로우 그래프를 사용해 수와 관련있는 계산을 하는
오픈소스 소프트웨어 라이브러리

Deep Learning with TensorFlow - Introduction to TensorFlow
https://www.youtube.com/watch?v=MotG3XI2qSs

Data Flow Graph
Graph: Computation Units
 - Node: Mathematical Operation
 - Edge: Multidemensionam data arrys(tensor)
         Communicated between them
   

tensor
A mathematical object analogous to but more general than a vector,
represented by an array of components that are functions of the coordinates of a space.

벡터와 유사하지만 더 일반적인 수학적 객체
공간 좌표 함수를 구성요소로 하는 배열로 표현

스칼라: 단지 크기만 있는 물리량
벡터: 방향과 크기가 있는 물리량

속력(스칼라)와 속도(벡터)의 차이
https://www.youtube.com/watch?v=dc9ifGNfe4I
'''

import tensorflow as tf

# add as a node to default graph
hello = tf.constant("hello tensorflow")

# start tf session
sess = tf.Session()

# run
print(sess.run(hello))
