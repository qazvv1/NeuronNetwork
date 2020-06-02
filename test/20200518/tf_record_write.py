import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


file_name = 'output.tfrecords'

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnist = input_data.read_data_sets('./mnist_data', one_hot=True, dtype=tf.uint8)
images = mnist.train.images  # shape (55000, 28*28*1)
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

writer = tf.python_io.TFRecordWriter(file_name)
for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={'pixels':_int64_feature(pixels),'label':_int64_feature(np.argmax(labels[index])), 'image_raw':_bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
writer.close()
    














