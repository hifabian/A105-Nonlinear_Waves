import numpy as np
from precision import *


def rk4(f, y0, h, t):
    k1 = h*f(y0, t)
    k2 = h*f(y0+half*k1, t+half*h)
    k3 = h*f(y0+half*k2, t+half*h)
    k4 = h*f(y0+k3, t+h)
    return y0+sixth*(k1+two*k2+two*k3+k4)


def ssm_integrator(f, y0, dt, T, filter, N = 0):
    flt_coef = [0.0625, -0.25, 0.375, -0.25, 0.0625]
    Aflt     = cst(0.1)

    y = y0; yres = []; t = zero; it = 0
    while t < T:
        y = rk4(f, y, dt, t)
        t += dt; it += 1
        if filter:
            for i in range(y.shape[1]):
                y[2:-2,i] -= Aflt*np.convolve(y[:,i], flt_coef,'same')[2:-2]
        if N and it % N == 0:
            yres.append(y)

    if len(yres) == 0:
        yres = y

    return t, np.array(yres)