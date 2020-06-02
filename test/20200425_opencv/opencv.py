import cv2
import matplotlib.pyplot as plt
from os import path
import numpy as np

#image = cv2.imread('D:\\python\\Neuron\\test\\20200425\\123.jpg')

# path1 = path.abspath(".")
# path1 = path.join(path1, "123.jpg")
# print(path1)

def imread(image):
    image = cv2.imread("123.jpg")   #opencv读取出的图片时BGR的
    # print(image.shape)
    # image = image[:, :, [2,1,0]]  #将读出的BGR转换成RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def image_show(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# cv2.imwrite("new_image.jpg",image)


#################### 图片平移 ##############
# image = imread("123.jpg")
# # 图片x右移250，y下移500
# M = np.float32([[1,0,250], [0,1,500]])
# switched = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
# image_show(switched)

#################### 图片旋转 ##############
image = imread("123.jpg")
w, h = image.shape[1], image.shape[0]
cX, cY = (w/2, h/2)#图片中心点
M = cv2.getRotationMatrix2D((cX,cY), 45, 1.0) # 逆时针转45度    1倍缩放
image = cv2.warpAffine(image, M, (w, h))
image_show(image)

####### 图片大小转换#######

image = cv2.resize(image, (150, 150))

