import tensorflow as tf


# input_files = ['./input_file_1', './input_file_2']
# data_set = tf.data.TextLineDataset(input_files)


# input_data = [1,2,3,4,5,6]
# data_set = tf.data.Dataset.from_tensor_slices(input_data)
# iterator = data_set.make_one_shot_iterator()
# x = iterator.get_next()
# y = x * x
# with tf.Session() as sess:
#     for i in range(len(input_data)):
#         print(sess.run(y))


def parser(record): # 解析TFrecord
    features = tf.parse_single_example(
        record,
        features={
            'feat1':tf.FixedLenFeature([], tf.int64)
            'feat2':tf.FixedLenFeature([], tf.int64)
        }
    )
    return features['feat1'], features['feat2']


input_files = ['./input_file_1', './input_file_2']
data_set = tf.data.TFRecordDataset(input_files)

data_set = data_set.map(parser)

iterator = data_set.make_one_shot_iterator()
feat1, feat2 = iterator.get_next()

with tf.Session() as sess:
    for _ in range(10):
        sess.run([feat1, feat2])


