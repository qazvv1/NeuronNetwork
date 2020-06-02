import tensorflow as tf
#tensorboard

a1 = tf.Variable(tf.random.uniform([3]), name='input1')
a2 = tf.constant([1.0, 2.0, 3.0], name='input2')

result = a1 + a2

writer = tf.summary.FileWriter('log', tf.get_default_graph())
writer.close()
