import tensorflow as tf 

with tf.device('/cpu:0'):
    a_cpu = tf.Variable(0, name='a_cpu')

with tf.device('/gpu:0'):
    a_gpu = tf.Variable(0, name='a_gpu')

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess.run(tf.initialize_all_variables())

