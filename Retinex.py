import cv2
import numpy as np



def single_scale_retinex(img, sigma):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return retinex


def multi_scale_retinex(img, sigma_list):
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += single_scale_retinex(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex



def color_restoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration



def retinex_process(img, sigma_list, G, b, alpha, beta):
    img = np.float64(img) + 1.0
    img_retinex = multi_scale_retinex(img, sigma_list)
    img_color = color_restoration(img, alpha, beta)
    img_retinex = G * (img_retinex * img_color + b)


    for i in range(img_retinex.shape[2]):
        img_retinex[:, :, i] = np.clip(img_retinex[:, :, i], 0, 255)
    img_retinex = np.uint8(img_retinex)

    return img_retinex


def main():
    # 读取图像
    img = cv2.imread('image1.jpg')

    # 尺度列表
    sigma_list = [15, 80, 250]
    # 增益参数
    G = 5.0
    # 偏置参数
    b = 25.0
    # 颜色恢复参数
    alpha = 125.0
    # 颜色恢复参数
    beta = 46.0

    # 进行图像增强
    img_retinex = retinex_process(img, sigma_list, G, b, alpha, beta)

    # 显示原始图像
    cv2.imshow('image1', img)
    # 显示增强后的图像
    cv2.imshow('Retinex', img_retinex)
    # 等待按键
    cv2.waitKey(0)
    # 保存增强后的图片
    cv2.imwrite('image2.jpg', img_retinex)


if __name__ == "__main__":
    main()
