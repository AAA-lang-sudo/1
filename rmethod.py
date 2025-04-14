import cv2
import numpy as np
import math
from scipy import signal
from skimage.util import img_as_float


def read_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    return img


def single_scale_retinex(img, sigma):
    retinex = np.log10(img + 1e-6) - np.log10(cv2.GaussianBlur(img + 1e-6, (0, 0), sigma))
    return retinex


def multi_scale_retinex(img, sigma_list):
    retinex = np.zeros_like(img, dtype=np.float64)
    for sigma in sigma_list:
        retinex += single_scale_retinex(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex


def color_restoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = beta * (np.log10(alpha * (img + 1e-6)) - np.log10(img_sum + 1e-6))
    return color_restoration


def retinex_process(img, sigma_list, G, b, alpha, beta):
    img = np.float64(img) + 1e-6
    img_retinex = multi_scale_retinex(img, sigma_list)
    img_color = color_restoration(img, alpha, beta)
    img_retinex = G * (img_retinex * img_color + b)

    for i in range(img_retinex.shape[2]):
        img_retinex[:, :, i] = np.clip(img_retinex[:, :, i], 0, 255)
    img_retinex = np.uint8(img_retinex)

    return img_retinex


def single_scale_retinex_with_gabor(img, sigma, theta, lambd, gamma, psi):
    img = img_as_float(img)
    gabor_kernel = cv2.getGaborKernel((31, 31), sigma, theta, lambd, gamma, psi)
    blurred = signal.convolve2d(img, gabor_kernel, mode='same')
    retinex = np.log10(img + 1e-6) - np.log10(np.abs(blurred) + 1e-6)
    return retinex


def multi_scale_retinex_with_gabor(img, sigma_list, theta, lambd, gamma, psi):
    retinex = np.zeros_like(img, dtype=np.float64)
    for sigma in sigma_list:
        for channel in range(img.shape[2]):
            retinex[:, :, channel] += single_scale_retinex_with_gabor(img[:, :, channel], sigma, theta, lambd, gamma,
                                                                      psi)
    retinex = retinex / len(sigma_list)
    return retinex


def retinex_gabor_process(img, sigma_list, G, b, alpha, beta, theta, lambd, gamma, psi):
    img = np.float64(img) + 1e-6
    img_retinex = multi_scale_retinex_with_gabor(img, sigma_list, theta, lambd, gamma, psi)
    img_color = color_restoration(img, alpha, beta)
    img_retinex = G * (img_retinex * img_color + b)

    for i in range(img_retinex.shape[2]):
        img_retinex[:, :, i] = np.clip(img_retinex[:, :, i], 0, 255)
    img_retinex = np.uint8(img_retinex)

    return img_retinex


def single_scale_retinex_with_laplacian(img, sigma):
    img = img_as_float(img)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    retinex = np.log10(img + 1e-6) - np.log10(np.abs(blurred - laplacian) + 1e-6)
    return retinex


def multi_scale_retinex_with_laplacian(img, sigma_list):
    retinex = np.zeros_like(img, dtype=np.float64)
    for sigma in sigma_list:
        for channel in range(img.shape[2]):
            retinex[:, :, channel] += single_scale_retinex_with_laplacian(img[:, :, channel], sigma)
    retinex = retinex / len(sigma_list)
    return retinex


def retinex_laplacian_process(img, sigma_list, G, b, alpha, beta):
    img = np.float64(img) + 1e-6
    img_retinex = multi_scale_retinex_with_laplacian(img, sigma_list)
    img_color = color_restoration(img, alpha, beta)
    img_retinex = G * (img_retinex * img_color + b)

    for i in range(img_retinex.shape[2]):
        img_retinex[:, :, i] = np.clip(img_retinex[:, :, i], 0, 255)
    img_retinex = np.uint8(img_retinex)

    return img_retinex


def single_scale_retinex_with_wavelet(img, sigma):
    import pywt
    img = img_as_float(img)
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    retinex = np.log10(img + 1e-6) - np.log10(np.abs(blurred - cA) + 1e-6)
    return retinex


def multi_scale_retinex_with_wavelet(img, sigma_list):
    retinex = np.zeros_like(img, dtype=np.float64)
    for sigma in sigma_list:
        for channel in range(img.shape[2]):
            retinex[:, :, channel] += single_scale_retinex_with_wavelet(img[:, :, channel], sigma)
    retinex = retinex / len(sigma_list)
    return retinex


def retinex_wavelet_process(img, sigma_list, G, b, alpha, beta):
    img = np.float64(img) + 1e-6
    img_retinex = multi_scale_retinex_with_wavelet(img, sigma_list)
    img_color = color_restoration(img, alpha, beta)
    img_retinex = G * (img_retinex * img_color + b)

    for i in range(img_retinex.shape[2]):
        img_retinex[:, :, i] = np.clip(img_retinex[:, :, i], 0, 255)
    img_retinex = np.uint8(img_retinex)

    return img_retinex


def show_images(original_img, enhanced_img, method_name):
    cv2.imshow(f'Original - {method_name}', original_img)
    cv2.imshow(f'Enhanced - {method_name}', enhanced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_images(original_img, enhanced_img, method_name):
    cv2.imwrite(f'original_{method_name}.jpg', original_img)
    cv2.imwrite(f'enhanced_{method_name}.jpg', enhanced_img)


def main():
    image_path = 'image1.jpg'
    img = read_image(image_path)
    if img is None:
        return

    sigma_list = [15, 80, 250]
    G = 5.0
    b = 25.0
    alpha = 125.0
    beta = 46.0

    theta = 0
    lambd = 10
    gamma = 0.5
    psi = 0

    img_retinex = retinex_process(img, sigma_list, G, b, alpha, beta)
    show_images(img, img_retinex, 'MS')
    save_images(img, img_retinex, 'MS')

    img_retinex_gabor = retinex_gabor_process(img, sigma_list, G, b, alpha, beta, theta, lambd, gamma, psi)
    show_images(img, img_retinex_gabor, 'MS with Gabor')
    save_images(img, img_retinex_gabor, 'MS with Gabor')

    img_retinex_laplacian = retinex_laplacian_process(img, sigma_list, G, b, alpha, beta)
    show_images(img, img_retinex_laplacian, 'MS with Laplacian')
    save_images(img, img_retinex_laplacian, 'MS with Laplacian')

    try:
        img_retinex_wavelet = retinex_wavelet_process(img, sigma_list, G, b, alpha, beta)
        show_images(img, img_retinex_wavelet, 'MS with Wavelet')
        save_images(img, img_retinex_wavelet, 'MS with Wavelet')
    except ImportError:
        print("检查环境")


if __name__ == "__main__":
    print("该方法时间较多")
    main()
