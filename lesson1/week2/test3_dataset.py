import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
# from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Load picture
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
# index = 22
# plt.imshow(train_set_x_orig[index])
# # plt.show()
# print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

# 计算图片数量m_train，图片像素num_px*num_px
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# 训练集和测试集变成 n*m_train 维
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

# standardize our dataset
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


# 
