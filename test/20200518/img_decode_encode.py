import matplotlib.pyplot as plt
import tensorflow as tf

# 解码
image_raw_data = tf.gfile.FastGFile('1.jpg', 'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    # print(img_data.eval())

    # plt.imshow(img_data.eval())
    # plt.show()

# 编码
    encoded_img = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile('2.jpg', 'wb') as f:
        f.write(encoded_img.eval())


