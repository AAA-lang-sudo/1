
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage


def read_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    img = img.astype(np.float32) / 255.0
    return img


def gamma_correction(img, gamma):
    return np.power(img, gamma)


def adaptive_gamma_correction(img, block_size=32, gamma_range=(0.1, 2.0)):
    height, width, _ = img.shape
    output = np.zeros_like(img)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = img[y:y + block_size, x:x + block_size]
            mean_intensity = np.mean(block)
            gamma = np.interp(mean_intensity, [0, 1], gamma_range)
            output[y:y + block_size, x:x + block_size] = gamma_correction(block, gamma)

    return output


def local_gamma_correction(img, radius=10, gamma=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = ndimage.gaussian_filter(gray, sigma=radius)
    local_mean = blurred[:, :, np.newaxis]
    local_gamma = gamma * (local_mean / np.max(local_mean))
    output = np.power(img, local_gamma)
    return output


def gamma_correction_with_lut(img, gamma):
    lut = np.array([((i / 255.0) ** gamma) * 255
                    for i in np.arange(0, 256)]).astype("uint8")
    img_uint8 = (img * 255).astype(np.uint8)
    corrected = cv2.LUT(img_uint8, lut)
    return corrected.astype(np.float32) / 255.0


def multi_scale_gamma_correction(img, scales=[1, 2, 4], gamma=0.5):
    height, width, _ = img.shape
    output = np.zeros_like(img)
    for scale in scales:
        scaled_img = cv2.resize(img, (width // scale, height // scale))
        corrected = gamma_correction(scaled_img, gamma)
        upscaled = cv2.resize(corrected, (width, height))
        output += upscaled
    output /= len(scales)
    return output


def plot_images(images, titles):
    num_images = len(images)

    rows = (num_images + 2) // 3
    plt.figure(figsize=(15, 5 * rows))
    for i in range(num_images):
        plt.subplot(rows, 3, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i], fontdict={'family': 'KaiTi', 'size': 10})
    plt.show()



def main():
    image_path = 'image1.jpg'
    img = read_image(image_path)
    if img is None:
        return

    gamma_values = [0.2, 0.5, 1.0, 1.5, 2.0]
    gamma_images = []
    titles = []

    for gamma in gamma_values:
        gamma_images.append(gamma_correction(img, gamma))
        titles.append(f'值: {gamma}')

    adaptive_img = adaptive_gamma_correction(img)
    gamma_images.append(adaptive_img)
    titles.append('自适应校正')

    local_img = local_gamma_correction(img)
    gamma_images.append(local_img)
    titles.append('局部校正')

    lut_img = gamma_correction_with_lut(img, 0.5)
    gamma_images.append(lut_img)
    titles.append('查找表校正')

    multi_scale_img = multi_scale_gamma_correction(img)
    gamma_images.append(multi_scale_img)
    titles.append('多尺度校正')

    plot_images(gamma_images, titles)

    cv2.imwrite("image_gamma_02.jpg", gamma_images[0] * 255)
    cv2.imwrite("image_gamma_05.jpg", gamma_images[1] * 255)
    cv2.imwrite("image_gamma_10.jpg", gamma_images[2] * 255)
    cv2.imwrite("image_gamma_15.jpg", gamma_images[3] * 255)
    cv2.imwrite("image_gamma_20.jpg", gamma_images[4] * 255)
    cv2.imwrite("image_adaptive.jpg", adaptive_img * 255)
    cv2.imwrite("image_local.jpg", local_img * 255)
    cv2.imwrite("image_lut.jpg", lut_img * 255)
    cv2.imwrite("image_multi_scale.jpg", multi_scale_img * 255)


if __name__ == "__main__":
    main()