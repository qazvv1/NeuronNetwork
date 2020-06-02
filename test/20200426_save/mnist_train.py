from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# mnist = input_data.read_data_sets("./mnist_data", one_hot=True)
# print("training data size: ", mnist.train.num_examples)
# print("validating data size: ", mnist.validation.num_examples)
# print("testing data size: ", mnist.test.num_examples)

# print(mnist.train.images[0])
# print(mnist.train.labels[0])

LAYERS = [784, 500, 10]
BATCH_SIZE = 100

LEARNING_RATE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEP = 30000
MOVING_AVERAGE_DECAY = 0.99


def inference(input, avg_class, w1, b1, w2, b2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input, w1) + b1)
        return tf.matmul(layer1, w2) + b2
    else:
        layer1 = tf.nn.relu(tf.matmul(input, avg_class.average(w1)) + avg_class.average(b1))
        return tf.matmul(layer1, avg_class.average(w2)) + avg_class.average(b2)

def train(mnist):
    x = tf.placeholder(tf.float32, [None, LAYERS[0]], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, LAYERS[2]], name='y_input')

    w1 = tf.Variable(tf.truncated_normal([LAYERS[0], LAYERS[1]], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[LAYERS[1]]))
    w2 = tf.Variable(tf.truncated_normal([LAYERS[1], LAYERS[2]], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[LAYERS[2]]))

    # 前向传播
    y = inference(x, None, w1, b1, w2, b2)

    # 划动平均
    global_step = tf.Variable(0, trainable=False)
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    average_y = inference(x, variable_average, w1, b1, w2, b2)

    # 交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # L2 正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(w1) + regularizer(w2)

    # 总损失
    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    train_op = tf.group(train_step, variable_average_op)

    # 计算正确率 ???
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}
        test_feed = {x:mnist.test.images, y_:mnist.test.labels}

        for i in range(TRAINING_STEP):
            if (i % 1000 == 0):
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps, validation accuracy using average model is %g" %(i, validate_acc))

            # sess.run(train_op, feed_dict={x:mnist.train.images[i * BATCH_SIZE % mnist.train.examples : i * BATCH_SIZE % mnist.train.examples + BATCH_SIZE], y_:x:mnist.train.labels[i * BATCH_SIZE % mnist.train.examples : i * BATCH_SIZE % mnist.train.examples + BATCH_SIZE]})
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x:xs, y_:ys})

        # 测试集上的准确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("test accuracy is %g" %(test_acc))

if (__name__ == "__main__") :
    mnist = input_data.read_data_sets("./mnist_data", one_hot=True)
    train(mnist)




