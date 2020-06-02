import tensorflow as tf

def print_eval(variables):
    for variable in variables:
        print(variable.eval())


v1 = tf.Variable(0, dtype=tf.float32, name="v1")
v2 = tf.Variable(0, dtype=tf.float32, name="v2")
result = v1 + v2

ema = tf.train.ExponentialMovingAverage(0.99)
average_op = ema.apply(tf.all_variables())

saver = tf.train.Saver()


with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    print_eval(tf.all_variables())
    sess.run(tf.assign(v1, 10))
    sess.run(tf.assign(v2, 1))
    sess.run(average_op)
    print_eval(tf.all_variables())

    saver.save(sess, "model.ckpt")




