# 伽马变换
# 在低照度图像增强中，伽马变换的原理是通过对图像的灰度值进行非线性变换，以提高图像的对比度和清晰度。
# 在低照度条件下，图像的亮度可能较低，细节不够清晰。通过应用伽马变换，可以调整图像的灰度级别，使得暗部细节更加突出，整体图像变得更清晰。
# 伽马变换可以增强图像的局部对比度，使图像更具有视觉吸引力和信息量。

import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('image1.jpg')
img = img.astype(np.float32) / 255.0


def gamma_correction(img, gamma):
    return np.power(img, gamma)


gamma = 0.5
img_gamma = gamma_correction(img, gamma)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('原图像', fontdict={'family': 'KaiTi', 'size': 10})

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_gamma, cv2.COLOR_BGR2RGB))
plt.title('伽马变换后图像', fontdict={'family': 'KaiTi', 'size': 10})
plt.show()

cv2.imwrite("image2.jpg", img_gamma * 255)  # 保存伽马变换后的图像