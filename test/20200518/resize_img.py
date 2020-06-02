import tensorflow as tf
import matplotlib.pyplot as plt


INPUT_FILE = '1.jpg' # size==(240, 231, 3)

image_raw_data = tf.gfile.FastGFile(INPUT_FILE, 'rb').read()
img_data = tf.image.decode_jpeg(image_raw_data)
# 将图片的0~255的int类型转换为0.0~1.0的实数类型
img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

# 重新设置图片尺寸resize
# method = 0~3
resized = tf.image.resize_images(img_data, [300, 300], method=0)

# 裁剪或填充resize
croped = tf.image.resize_image_with_crop_or_pad(img_data, 100, 100)
padded = tf.image.resize_image_with_crop_or_pad(img_data, 600, 600)

# 固定比例resize
central_croped = tf.image.central_crop(img_data, 0.5)

# 裁剪或填充指定区域（上面的是以中心点）
# tf.image.crop_to_bounding_box
# tf.image.pad_to_bounding_box

# 将图像上下翻转
flipped = tf.image.flip_up_down(img_data)
# 将图像左右翻转
flipped = tf.image.flip_left_right(img_data)
# 将图像沿对角线翻转
transposed = tf.image.transpose_image(img_data)

# 50%几率 下翻转圈像。
flipped= tf.image.random_flip_up_down(img_data)
# 50%几率 左右转圈像。
flipped= tf.image.random_flip_left_right(img_data)

# 亮度 -0.5
adjusted = tf.image.adjust_brightness(img_data, -0.5)
# 调整后，将每个点限制在0.0~1.0之间
adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)

# 在 [-max_delta, max_delta) 的范围随机调整图像的亮度 -1.0~1.0
adjusted =  tf.image.random_brightness(image, max_delta)

# 对比度调整为0.5倍
adjusted= tf.image.adjust_contrast(img_data, 0.5)
# 随机调整对比度
adjusted = tf.image.random_contrast(img_data, lower, upper)

# 色相 加0.1
adjusted = tf.image.adjust_hue(img_data, 0.1)
# 随机调整色相
adjusted = tf.image.random_hue(img_data, max_delta)

# 饱和度
adjusted = tf.image.adjust_saturation(img_data, -5)
adjusted = tf.image.random_saturation(img_data, lower, upper)

# 标准化    均值=0  方差=1
adjusted = tf.image.per_image_standardization(img_data)


############## 标注框 ##################
img_data = tf.image.resize_images(img_data, [180,267], method=1)
# tf.image.draw_bounding_boxes 输入为实数，四维[batch, w, h, channels]
batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]]) # 相对比例(左下点和右上点坐标)
result = tf.image.draw_bounding_boxes(batched, boxes)

# 随机截取有用信息
boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(img_data), boxes, min_object_covered=0.4) # 至少取某个box中40%的内容
batched = tf.image.draw_bounding_boxes(img_data, bbox_for_draw) # 在原图上可视化
distorted_image = tf.slice(img_data, begin, size) # 随机截取图的可视化





with tf.Session() as sess:
    # print(sess.run(tf.shape(img_data)))

    # plt.imshow(croped.eval())
    # plt.imshow(padded.eval())
    plt.imshow(central_croped.eval())
    plt.show()





