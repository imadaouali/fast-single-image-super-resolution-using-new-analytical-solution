# %%

import matplotlib.pyplot as plt
from skimage import io
import os
from image_super_resolution.analytical import *
from image_super_resolution.utils import gauss2D_Kernel, blockMM

PATH = r'C:/Users/essou/Desktop/my_projects/image-super-resolution/images/'
B = gauss2D_Kernel(shape=(20, 20), sigma=3)


def main(img_path, gaussion_kernel=B):
    img = io.imread(img_path)
    # High resolution image
    refl = img

    # Blurring
    print(r'Performing blurring : $y=Bx$--------------------------------------------', end='', sep='')
    FB, FBC, F2B, Bx = HXconv(refl, gaussion_kernel)
    print("[COMPLETED]")

    N = refl.shape[0]*refl.shape[1]
    Psig = np.linalg.norm(Bx, 'fro')**2/N
    BSNRdb = 40
    sigma = np.linalg.norm(Bx-np.mean(Bx), 'fro')/np.sqrt(N*10**(BSNRdb/10))
    m, n = refl.shape

    # Additive noise
    print(r'Adding noise : $y=Bx+n$--------------------------------------------', end='', sep='')
    # Here we create the noisy low resolution image from the HR image
    y = Bx + sigma*np.random.randn(m, n)
    FY = fft2(y)
    print("[COMPLETED]")

    d = 2
    dr, dc = d, d
    Nb = dr*dc

    # Decimation
    print('Performing decimation : --------------------------------------------', end='', sep='')
    y = y[::dr, ::dc]
    reflp = refl[::dr, ::dc]
    print("[COMPLETED]")

    # Bicubic interpolation
    print('Performing bicubic interpolation: --------------------------------------------', end='', sep='')
    t = time()
    # Here we compute the bicubic interpolation
    yinp = cv2.resize(
        y, (d*y.shape[1], d*y.shape[0]), interpolation=cv2.INTER_CUBIC)
    dt = time() - t
    print(f"[COMPLETED] in {dt}")

    # L2 norm Super-resolution with Analytical solution
    taup = 2e-3
    tau = taup*sigma**2
    nr, nc = y.shape
    m = nr*nc
    nrup = nr*d
    ncup = nc*d

    # Analytical super resolution algorithm
    xp = yinp
    STy = np.zeros((nrup, ncup))
    STy[::d, ::d] = y
    FR = FBC*fft2(STy) + fft2(2*tau*xp)

    print('Performing super resolution algorithm: --------------------------------------------', end='', sep='')
    t = time()
    Xest_analytic = INVLS(FB, FBC, F2B, FR, 2*tau, Nb, nr, nc)
    dt = time() - t
    print(f"[COMPLETED] in {dt}")

    # Metrics
    print("#########################################################################################")
    print("Main metrics")
    print(" - ISNR : " + str(ISNR(yinp, refl, Xest_analytic)))
    print(" - PSNR : " + str(PSNR(refl, Xest_analytic)))
    print(" - PSNR with bicubic: " + str(PSNR(yinp, refl)))
    print("#########################################################################################")

    # Display
    f, ax = plt.subplots(2, 1, figsize=(5, 15))
    ax[0].imshow(y)
    ax[0].title.set_text('Low Resolution Image')
    ax[1].imshow(Xest_analytic)
    ax[1].title.set_text('High Resolution Image with FSR Analytical')
    plt.tight_layout()


if __name__ == "__main__":
    img_path = os.path.join(PATH, 'lena.bmp', '')
    main(img_path)


# %%
