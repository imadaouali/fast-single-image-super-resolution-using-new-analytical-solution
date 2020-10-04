import numpy as np
from numpy.fft import fft, ifft, fftshift, fft2, ifft2
from scipy.ndimage import gaussian_filter
from skimage import io
from scipy.ndimage.filters import convolve
import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from time import time


def gauss2D_Kernel(shape=(3, 3), sigma=0.5):
    """creates a gaussian blurring kernel

    Args:
        shape (tuple, optional): shape of kernel. Defaults to (3,3).
        sigma (float, optional): standard deviation of blurring kernel. Defaults to 0.5.

    Returns:
        np.array: bluuring kernel
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def segmented_process(M, blk_size, fun=None):
    """processes the image M by applying the function fun to each distinct block of size blk_size and concatenating the results into the output matrix, M_res.

    Args:
        M (np.array): input matrix
        blk_size (tuple): size of blocks to process
        fun (lambda function, optional): function to apply. Defaults to None.

    Returns:
        np.array: processed matrix
    """
    rows = []
    for i in range(0, M.shape[0], blk_size[0]):
        cols = []
        for j in range(0, M.shape[1], blk_size[1]):
            cols.append(fun(M[i:i+blk_size[0], j:j+blk_size[1]]))
        rows.append(np.concatenate(cols, axis=1))
    M_res = np.concatenate(rows, axis=0)
    return M_res


def blockMM(nr, nc, Nb, x1):
    """decimate image

    Args:
        nr (int): size on x_axis of low_resolution image
        nc (int): size on y_axis of low_resolution
        Nb (int): scale factor
        x1 (np.array): input high_resolution image

    Returns:
        np.array: decimated image
    """
    def myfun(block): return block.reshape((1, nr*nc))
    x1 = segmented_process(x1, blk_size=(nr, nc), fun=myfun)
    x1 = x1.reshape((Nb, nr*nc))
    x1 = x1.sum(axis=0)
    x = x1.reshape((nr, nc))

    return x


def ISNR(y, x, x_hat):
    """compute signal-to-noise ratio
    """
    return 10*np.log10(np.linalg.norm(x-y, 'fro')**2/np.linalg.norm(x-x_hat, 'fro')**2)


def PSNR(img1, img2):
    """computes peak signal-tonoise ratio

    Args:
        img1 (np.array): image
        img2 (np.array): image

    Returns:
        float: peak signal-tonoise ratio
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
