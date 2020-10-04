#%%
import matplotlib.pyplot as plt
from skimage import io
import os
from image_super_resolution.analytical import *
from image_super_resolution.utils import gauss2D_Kernel, blockMM
from image_super_resolution.iterative import *

PATH = r'C:/Users/essou/Desktop/my_projects/image-super-resolution/images/'
B = gauss2D_Kernel(shape=(20, 20), sigma=3)


def main(img_path, gaussion_kernel=B):
    refl = io.imread(img_path)
    # Blurring and decimation

    print('Performing decimation and blurring: --------------------------------------------', end='', sep='')

    FB, FBC, F2B, Bx = HXconv(refl, B)
    N = len(refl)*len(refl[0])
    Psig = np.linalg.norm(Bx, 'fro')**2/N
    BSNRdb = 40
    sigma = np.linalg.norm(Bx-np.mean(Bx), 'fro')/np.sqrt(N*10**(BSNRdb/10))
    m, n = refl.shape
    y = Bx + sigma*np.random.randn(m, n)
    FY = fft2(y)

    d = 2
    dr, dc = d, d
    Nb = dr*dc

    y = y[::dr, ::dc]
    reflp = refl[::dr, ::dc]
    yinp = cv2.resize(
        y, (d*y.shape[1], d*y.shape[0]), interpolation=cv2.INTER_CUBIC)

    taup = 2e-3
    tau = taup*sigma**2
    nr, nc = y.shape
    m = nr*nc
    nrup = nr*dr
    ncup = nc*dc

    xp = yinp
    STy = np.zeros((nrup, ncup))
    STy[0:-1:d, 0:-1:d] = y
    FR = FBC*fft2(STy) + fft2(2*tau*xp)
    print(f'[COMPLETED]')

    admm(base_image=refl, X=yinp, FB=FB, y=y, xp=xp, STy=STy, FBC=FBC,
         F2B=F2B, tolA=1e-4, tau=tau, mu=0.01, maxiter=100, dr=dr, dc=dc)


if __name__ == "__main__":
    img_path = os.path.join(PATH, 'lena.bmp', '')
    main(img_path)


# %%
