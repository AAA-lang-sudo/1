
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.exposure import equalize_adapthist, equalize_hist
from skimage.util import img_as_ubyte


def read_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    return img


def simple_global_histogram_equalization(img):
    B, G, R = cv2.split(img)
    b = cv2.equalizeHist(B)
    g = cv2.equalizeHist(G)
    r = cv2.equalizeHist(R)
    equal_img = cv2.merge((b, g, r))
    return equal_img


def global_histogram_equalization_skimage(img):
    img_float = img / 255.0
    equalized = np.zeros_like(img_float)
    for i in range(img_float.shape[2]):
        equalized[:, :, i] = equalize_hist(img_float[:, :, i])
    equalized = img_as_ubyte(equalized)
    return equalized


def adaptive_histogram_equalization_skimage(img):
    img_float = img / 255.0
    equalized = np.zeros_like(img_float)
    for i in range(img_float.shape[2]):
        equalized[:, :, i] = equalize_adapthist(img_float[:, :, i], clip_limit=0.03)
    equalized = img_as_ubyte(equalized)
    return equalized


def clahe_equalization(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    B, G, R = cv2.split(img)
    b = clahe.apply(B)
    g = clahe.apply(G)
    r = clahe.apply(R)
    equal_img = cv2.merge((b, g, r))
    return equal_img


def compute_histogram(img, channel=0):
    return cv2.calcHist([img], [channel], None, [256], [0, 256])


def plot_histograms(original_img, enhanced_img, title_original, title_enhanced):
    hist_original = compute_histogram(original_img)
    hist_enhanced = compute_histogram(enhanced_img)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(hist_original, 'b')
    plt.title(title_original, fontdict={'family': 'KaiTi', 'size': 10})
    plt.subplot(1, 2, 2)
    plt.title(title_enhanced, fontdict={'family': 'KaiTi', 'size': 10})
    plt.plot(hist_enhanced, 'b')
    plt.show()


def show_images(original_img, enhanced_img, title_original, title_enhanced):
    cv2.imshow(title_original, original_img)
    cv2.imshow(title_enhanced, enhanced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(img, filename):
    cv2.imwrite(filename, img)


def contrast_stretching(img):
    min_val = np.min(img)
    max_val = np.max(img)
    stretched = (img - min_val) / (max_val - min_val) * 255
    stretched = stretched.astype(np.uint8)
    return stretched


def power_law_transform(img, gamma=1.0):
    img_float = img.astype(np.float32) / 255.0
    transformed = np.power(img_float, gamma)
    transformed = (transformed * 255).astype(np.uint8)
    return transformed


def main():
    image_path = "image1.jpg"
    img = read_image(image_path)
    if img is None:
        return

    simple_global_equal = simple_global_histogram_equalization(img)
    plot_histograms(img, simple_global_equal, '原图B通道', '简单全局均衡化后B通道')
    show_images(img, simple_global_equal, '原图', '简单全局均衡化后图像')
    save_image(simple_global_equal, "simple_global_equal.jpg")

    global_equal_skimage = global_histogram_equalization_skimage(img)
    plot_histograms(img, global_equal_skimage, '原图B通道', 'skimage全局均衡化后B通道')
    show_images(img, global_equal_skimage, '原图', 'skimage全局均衡化后图像')
    save_image(global_equal_skimage, "global_equal_skimage.jpg")

    adaptive_equal_skimage = adaptive_histogram_equalization_skimage(img)
    plot_histograms(img, adaptive_equal_skimage, '原图B通道', 'skimage自适应均衡化后B通道')
    show_images(img, adaptive_equal_skimage, '原图', 'skimage自适应均衡化后图像')
    save_image(adaptive_equal_skimage, "adaptive_equal_skimage.jpg")

    clahe_equal = clahe_equalization(img)
    plot_histograms(img, clahe_equal, '原图B通道', 'CLAHE均衡化后B通道')
    show_images(img, clahe_equal, '原图', 'CLAHE均衡化后图像')
    save_image(clahe_equal, "clahe_equal.jpg")

    stretched_img = contrast_stretching(img)
    plot_histograms(img, stretched_img, '原图B通道', '对比度拉伸后B通道')
    show_images(img, stretched_img, '原图', '对比度拉伸后图像')
    save_image(stretched_img, "stretched_img.jpg")

    gamma_values = [0.5, 1.5, 2.0]
    for gamma in gamma_values:
        power_transformed = power_law_transform(img, gamma)
        plot_histograms(img, power_transformed, '原图B通道', f'幂律变换(gamma={gamma})后B通道')
        show_images(img, power_transformed, '原图', f'幂律变换(gamma={gamma})后图像')
        save_image(power_transformed, f"power_transformed_gamma_{gamma}.jpg")


if __name__ == "__main__":
    main()