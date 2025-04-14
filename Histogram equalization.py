# 直方图均衡化
# 直方图均衡化的基本原理是对灰度值图像的各个像素的灰度级进行再分配，使得图像整体像素分布的灰度平均化，从而增强整体的对比度。
# 该方法通过对图像的像素值进行变换，拉伸或压缩原始图像的像素值范围，使得图像的直方图更加平坦，从而达到图像的增强效果。
# 该方法的使用可以提高图像像素的对比度、去除光照不均匀，适用于灰度图像和低照度图像。
# 然而，直方图均衡化容易导致图像过度增强或细节丢失的问题，对于低照度图像的增强效果有限，对于噪声较多的图像，直方图均衡化可能会增强噪声，需结合去噪处理。


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("image1.jpg")
B, G, R = cv2.split(img)  # get single 8-bits channel
b = cv2.equalizeHist(B)
g = cv2.equalizeHist(G)
r = cv2.equalizeHist(R)
equal_img = cv2.merge((b, g, r))  # merge it back

hist_b = cv2.calcHist([equal_img], [0], None, [256], [0, 256])
hist_B = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.subplot(1, 2, 1)
plt.plot(hist_B, 'b')
plt.title('原图B通道的直方图', fontdict={'family': 'KaiTi', 'size': 10})
plt.subplot(1, 2, 2)
plt.title('均衡化后B通道的直方图', fontdict={'family': 'KaiTi', 'size': 10})
plt.plot(hist_b, 'b')
plt.show()
cv2.imwrite("image.jpg", equal_img)

cv2.imshow("orj", img)
cv2.imshow("equal_img", equal_img)
cv2.waitKey(0)