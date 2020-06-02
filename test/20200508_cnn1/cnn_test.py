import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


w1 = tf.get_variable("w1", [5,5,3,16], initializer=tf.truncated_normal_initializer(stddev=0.1))
b1 = tf.get_variable("b1", [16], initializer=tf.constant_initializer(stddev=0.1))

conv = tf.nn.conv2d(X, w1, strides=[1,2,2,1], padding="SAME")






