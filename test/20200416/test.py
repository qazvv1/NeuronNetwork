import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a", dtype=tf.float32)
b = tf.constant([1.0, 2.0], name="b")

result = a * b

with tf.Session() as sess:
    print(sess.run(result))
    print(result.eval())

sess = tf.Session()
print(sess.run(result))
print(result.eval(session=sess))

