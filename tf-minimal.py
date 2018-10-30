import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import numpy as np

with tf.Session() as sess:
   dimension = 1
   X = tf.placeholder(tf.float32, [None, dimension], name="X")
   W = tf.Variable(tf.random_normal([dimension, 100], stddev=0.01))
   b = tf.Variable(tf.zeros([100]))
   h1 = tf.nn.relu(tf.matmul(X, W) + b)

   W2 = tf.Variable(tf.random_normal([100, 50], stddev=0.01))
   b2 = tf.Variable(tf.zeros([50]))
   h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

   W3 = tf.Variable(tf.random_normal([50, 1], stddev=0.01))
   b3 = tf.Variable(tf.zeros([1]))
   y = (tf.matmul(h2, W3) + b3)

   vy = tf.identity(y, name="vy")

   Y = tf.placeholder(tf.float32, [None, dimension], name="Y")

   cost = tf.reduce_mean(tf.squared_difference(vy, Y))
   optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
   init = tf.global_variables_initializer()

   sess.run(init)
   cap = 50
   for epoch in range(2000):
      sx = np.random.randint(cap, size=(100, 1))
      # sx = np.random.rand(100,1)
      sy = np.sqrt(sx)
      op, c = sess.run([optimizer, cost], feed_dict={X: sx, Y: sy})
      if epoch % 100 == 0:
         print ("Epoch:", '%04d' % (epoch + 1), "cost=", "%.03f" % c)

   # sx = np.random.rand(10,1)
   sx = np.random.randint(cap, size=(1, 1))
   sy = np.sqrt(sx)
   print ("Input")
   print (sx)
   print ("Expected Output")
   print (sy)
   print ("Predicted Output")
   print (sess.run(vy, feed_dict={X: sx, Y: sy}))
   print ("Error")
   print (sess.run(cost, feed_dict={X: sx, Y: sy}))

   frozen_graph = convert_variables_to_constants(sess, sess.graph_def, ['vy', 'Y'])
   tf.train.write_graph(frozen_graph, './', 'graph2.pb', as_text=False)


