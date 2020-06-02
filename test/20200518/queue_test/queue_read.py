import tensorflow as tf


files = tf.train.match_filenames_once('./queue_test/data.tfrecords-*')
filename_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example, features={
    'i':tf.FixedLenFeature([], tf.int64),
    'j':tf.FixedLenFeature([], tf.int64)
})


# with tf.Session() as sess:
#     tf.local_variables_initializer().run()
#     print(sess.run(files))

#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess, coord)
#     for _ in range(6):
#         # print(sess.run(features))
#     coord.request_stop()
#     coord.join(threads)


example, label = features['i'], features['j']
batch_size = 3
capacity = 1000 + 3 * batch_size

example_batch, label_batch = tf.train.batch([example, label], batch_size, capacity=capacity)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    for _ in range(2):
        print(sess.run([example_batch, label_batch]))
    coord.request_stop()
    coord.join(threads)
