import numpy as np
from numpy.fft import fft, ifft, fftshift, fft2, ifft2
from scipy.ndimage import gaussian_filter
from skimage import io
from scipy.ndimage.filters import convolve
import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from time import time
from image_super_resolution.utils import *


def HXconv(x, B):
    """returns the blurred image and other objects

    Args:
        x (np.array): High resolution image
        B (np.array): Blurring kernel

    Returns:
        BF (np.array): The Fourier Transform of the Blurring Kernel 
        BFC (np.array): Conj(BF) 
        B2F (np.array):abs(BF)**2 
        y (np.array): The blurred image 
    """
    m, n = x.shape
    m0, n0 = B.shape

    Bpad = np.zeros((m, n))
    Bpad[int((m-m0)/2):int((m+m0)/2), int((n-n0)/2):int((n+n0)/2)] = B
    Bpad = fftshift(Bpad)

    BF = fft2(Bpad)
    BCF = np.conj(BF)
    B2F = np.abs(BF)**2

    y = np.real(ifft2(BF*fft2(x)))

    return BF, BCF, B2F, y


def INVLS(FB, FBC, F2B, FR, tau, Nb, nr, nc):
    """computes HR image

    Args:
        FB (np.array): Fourier transform of the blurring kernel B 
        FBC (np.array): conj(FB) 
        F2B (np.array): abs(FB)**2 
        FR (np.array): Fourier transform of R 
        Nb (int): scale factor Nb = dr*dc 
        nr,nc (int,int): size of the observation 

    Returns:
        np.array: High resolution image
    """
    x1 = FB*FR
    FBR = blockMM(nr, nc, Nb, x1)
    invW = blockMM(nr, nc, Nb, F2B)
    invWBR = FBR/(invW + tau*Nb)
    def myfun(block): return block*invWBR
    FCBinvWBR = segmented_process(FBC, blk_size=(nr, nc), fun=myfun)
    FX = (FR-FCBinvWBR)/(tau)
    Xest = np.real(ifft2(FX))

    return Xest
