import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


num_shards = 2 # 写入的文件数量
instance_per_shard = 2 # 每个文件的数据量

for i in range(num_shards):
    file_name = ('./queue_test/data.tfrecords-%.5d-of%.5d' % (i, num_shards))
    writer = tf.python_io.TFRecordWriter(file_name)

    for j in range(instance_per_shard):
        example = tf.train.Example(features=tf.train.Features(feature={
            'i':_int64_feature(i),
            'j':_int64_feature(j)}))
        writer.write(example.SerializeToString())
    writer.close()
