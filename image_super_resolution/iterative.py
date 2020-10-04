import matplotlib.pyplot as plt
from skimage import io
import os
from image_super_resolution.analytical import *
from image_super_resolution.utils import gauss2D_Kernel, blockMM


def admm(base_image, X, FB, y, xp, dr, dc, STy, FBC, F2B, tolA=1e-4, mu=0.01, tau=2e-3, maxiter=100, plot_result=True):
    """ admm iterative algorithm
    """

    BX = ifft2(FB*fft2(X))
    resid = ifft2(y - BX[::dr, ::dc])
    prev_f = 0.5*(np.linalg.norm(resid, 'fro')**2 + tau*np.sum(np.abs(X-xp)))

    nr, nc = y.shape

    objective = np.zeros(maxiter)
    objective[0] = prev_f
    ISNR_admmSet = np.zeros(maxiter)
    PSNR_admmSet = np.zeros(maxiter)

    nrup = nr*dr
    ncup = nc*dc
    STytemp = np.ones((nrup, ncup))
    STytemp[::dr, ::dc] = y
    ind1 = np.where(STytemp-STy == 1)
    ind2 = np.where(STytemp-STy == 0)

    print(r'Performing ADMM recovery : --------------------------------------------', end='', sep='')
    t0 = time()
    gam = tau/mu
    X = xp
    U = X
    D = np.zeros((nrup, ncup))

    for i in range(maxiter-1):
        # update X
        V = U-D
        Fb = FBC*fft2(V) + fft2(2*gam*xp)
        FX = Fb/(F2B + 2*gam)
        X = ifft2(FX)
        # update u
        rr = mu*(ifft2(FB*FX) + D)
        temp1 = rr[ind1]/mu
        temp2 = (rr[ind2] + STy[ind2])/(1+mu)
        U[ind1] = temp1
        U[ind2] = temp2
        # update d
        D = D + (ifft2(FB*FX)-U)

        ISNR_admmSet[i] = ISNR(xp, base_image, X)
        PSNR_admmSet[i] = PSNR(base_image, X)
        BX = ifft2(FB*fft2(X))
        resid = ifft2(y - BX[::dr, ::dc])
        objective[i+1] = 0.5*(np.linalg.norm(resid, 'fro')
                              ** 2 + tau*np.sum(np.abs(X-xp)))

        if np.abs(objective[i+1]-objective[i])/objective[i] < tolA:
            break

    dt = time() - t0
    print(f"[COMPLETED] in {dt}")

    # recovered pictures
    X_recovery = np.real(X)

    # metrics
    print("ISNR : " + str(ISNR(xp, base_image, X_recovery)))
    print("PSNR : " + str(PSNR(base_image, X_recovery)))

    # plot results
    if plot_result:
        plt.rcParams['axes.grid'] = True
        f, ax = plt.subplots(2, 2, figsize=(15, 15))
        ax[0, 0].imshow(X_recovery, cmap='gray')
        ax[0, 0].title.set_text('Low Resolution Image')
        ax[0, 1].imshow(X_recovery, cmap='gray')
        ax[0, 1].title.set_text('High Resolution Image with admm algorithm')
        ax[1, 0].plot(range(maxiter-1), ISNR_admmSet[:-1],
                      label='ISNR - BARBARA', color='r')
        ax[1, 0].title.set_text('ISNR in function of number of iterations')
        ax[1, 1].plot(range(maxiter-1), PSNR_admmSet[:-1],
                      label='ISNR - BARBARA', color='r')
        ax[1, 1].title.set_text('PSNR in function of number of iterations')
        f.subplots_adjust(hspace=0.5)
        # plt.tight_layout()
        plt.show()

    return None
