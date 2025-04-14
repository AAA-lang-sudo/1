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
