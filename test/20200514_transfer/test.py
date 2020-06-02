import tensorflow.contrib.slim as slim
import tensorflow as tf

for var in slim.get_model_variables():
    print(var)
    print('1')
    print('2')
    print('3')

tf.GraphKeys.TRAINABLE_VARIABLES
