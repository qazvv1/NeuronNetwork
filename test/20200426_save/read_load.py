import tensorflow as tf

## save
# v1 = tf.Variable(tf.constant(1.0, shape=[2]), name="v1")
# saver = tf.train.Saver()


## load
saver = tf.train.import_meta_graph("model.ckpt.meta")

with tf.Session() as sess:
    ## save
    # sess.run(tf.initialize_all_variables())
    # saver.save(sess, "model.ckpt")

    ## load
    saver.restore(sess, "model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("v1:0")))


