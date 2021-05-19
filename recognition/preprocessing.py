import numpy as np
import cv2


def normalize_input(image):
    """
    normalize image for face net input layer
    :param image:
    :return:
    """
    image = np.array(image)
    mean = np.mean(image)
    std = np.std(image)
    std_adj = np.maximum(std, 1.0 / np.sqrt(image.size))
    image = np.multiply(np.subtract(image, mean), 1 / std_adj)
    return image


def normalize_faces(img: np.ndarray) -> np.ndarray:
    mean = np.mean(img, axis=(1, 2, 3))
    std = np.std(img, axis=(1, 2, 3))
    std_adj = np.maximum(std, 1.0 / np.sqrt(img[0].size))
    return np.multiply(np.subtract(img, mean.reshape((-1, 1, 1, 1))), 1 / std_adj.reshape(-1, 1, 1, 1))


def cvt_to_gray(im):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = np.dstack([im, im, im])
    return im
