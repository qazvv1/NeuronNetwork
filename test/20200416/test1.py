import tensorflow as tf
from numpy.random import RandomState

mini_batch_size = 8
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, [None, 2], name='x_input')
y = tf.placeholder(tf.float32, [None, 1], name='y_input')

a = tf.matmul(x, w1)
y_hat = tf.matmul(a, w2)

loss = -tf.reduce_mean(y * tf.log(tf.clip_by_value(y_hat, 1e-10, 1.0)))

train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

X = RandomState(1).rand(128, 2) # (128, 2)维的训练集数据X
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]    # X对应的Y

############# train ###########
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for i in range(5000):   # 训练轮数
        start = i * mini_batch_size % 128
        end = min(start + mini_batch_size, 128)
        sess.run(train_step, feed_dict={x : X[start:end],y : Y[start:end]})
        if i % 1000 == 0:   # 每训练1000次，就把所有X扔进模型，看看算出来的交叉熵是多少
            total_cross_entropy = sess.run(loss, feed_dict={x:X, y:Y})
            print("After %d training steps, cross entropy on all data is %g" % (i, total_cross_entropy))
    


