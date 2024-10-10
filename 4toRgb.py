from PIL import Image
import matplotlib.pyplot as plt
#
# img_path = "./HyperImage_data/6001.png"
# save_path = './HyperImage_data/6003.png'
#
# img = Image.open(img_path)
# img = img.convert("RGB")
# img.save(save_path)

import cv2

img_path = "/root/autodl-tmp/mssgu/data/ff1024/821_512_gt.png"
save_path = '/root/autodl-tmp/mssgu/data/ff1024/821_512_gt.png'
# img_path = "E:/gcn/MSSGU-UNet-main-DM/HyperImage_data/821_510.png"
# save_path = 'E:/gcn/MSSGU-UNet-main-DM/HyperImage_data/821_510_rgb.png'

# 其实使用的方法非常简单，就是使用cv2.imread()读取四通道图片
# 图片格式会自动转为三通道格式。
img = cv2.imread(img_path)
#
# # 再通过cv2.imwrite()直接保存，图片就保存为三通道
# # 之后用其他方式再读取就是三通道格式
cv2.imwrite(save_path, img)


import numpy as np
from PIL import Image
import imageio

# 24转8
# 根据路径读取图像
# img = Image.open("data/SegmentationClass/931.png")
#
# # 转换为numpy数组
# img_arr = np.array(img)
#
# # 将RGB图像转换为灰度图像
# gray_img = np.mean(img_arr, axis=-1) # 取RGB三通道的平均值
#
# # 将灰度图像转换为8位图像
# gray_img = gray_img.astype(np.uint8) # 转换为8位整型数据类型
#
# # 将图像保存为8位图像
# imageio.imwrite("data/SegmentationClass/931.png", gray_img)

#
# 8转24
# img = Image.open('/root/autodl-tmp/mssgu/data/ff1024/ff1024_gt.png')
# img = img.convert('RGB')
# img.save('/root/autodl-tmp/mssgu/data/ff1024/ff1024_gt.png')