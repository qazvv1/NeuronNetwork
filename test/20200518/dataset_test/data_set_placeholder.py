import tensorflow as tf


def parser(record):  # 解析TFrecord
    features = tf.parse_single_example(
        record,
        features={
            'feat1': tf.FixedLenFeature([], tf.int64)
            'feat2': tf.FixedLenFeature([], tf.int64)
        }
    )
    return features['feat1'], features['feat2']


input_files = tf.placeholder(tf.string)
data_set = tf.data.TFRecordDataset(input_files)
data_set = data_set.map(parser)

iterator = tf.data.make_initializable_iterator(data_set)
feat1, feat2 = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer,
             feed_dict={input_files: [file1, file2]})

    while True:
        try:
            sess.run(feat1, feat2)
        except tf.errors.OutOfRangeError:
            break
