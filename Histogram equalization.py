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
