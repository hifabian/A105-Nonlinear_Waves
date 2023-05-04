import numpy as np
from precision import *


def second_order(dx, periodic):

    c_dfdx = half/dx
    c_d2dfx2 = one/(dx*dx)
    if not periodic:
        def dfdx(f):
            df = c_dfdx*(np.roll(f, -1, axis=0)-np.roll(f, 1, axis=0))
            df[0] = c_dfdx*(-three*f[0]+four*f[1]-f[2])
            df[-1] = c_dfdx*(three*f[-1]-four*f[-2]+f[-3])
            return df
        def d2fdx2(f):
            df = c_d2dfx2*(np.roll(f, -1, axis=0)+np.roll(f, 1, axis=0)-two*f)
            df[0] = c_d2dfx2*(two*f[0]-five*f[1]+four*f[2]-f[3])
            df[-1] = c_d2dfx2*(two*f[-1]-five*f[-2]+four*f[-3]-f[-4])
            return df
    else:
        def dfdx(f):
            df = c_dfdx*(np.roll(f, -1, axis=0)-np.roll(f, 1, axis=0))
            return df
        def d2fdx2(f):
            df = c_d2dfx2*(np.roll(f, -1, axis=0)+np.roll(f, 1, axis=0)-two*f)
            return df
    return dfdx, d2fdx2


def fourth_order(dx, periodic):
    c_dfdx = twelth/dx
    c_d2dfx2 = twelth/(dx*dx)
    if not periodic:
        def dfdx(f):
            df = c_dfdx*(np.roll(f, 2, axis=0)-eight*np.roll(f, 1, axis=0)
                         +eight*np.roll(f, -1, axis=0)-np.roll(f, -2, axis=0))
            df[:2] = c_dfdx*(-3*f[4:6]-25*f[:2]-36*f[2:4]+48*f[1:3]+16*f[3:5])
            df[-2:] = c_dfdx*(3*f[-6:-4]+25*f[-2:]+36*f[-4:-2]-48*f[-3:-1]-16*f[-5:-3])
            return df
        def d2fdx2(f):
            df = c_d2dfx2*(16*np.roll(f, 1, axis=0)+16*np.roll(f, -1, axis=0)
                           -30*f-np.roll(f, 2, axis=0)-np.roll(f, -2, axis=0))
            # Second-order at boundary, because other doesn't seem to work...
            df[:2] = 12*c_d2dfx2*(two*f[:2]-five*f[1:3]+four*f[2:4]-f[3:5])
            df[-2:] = 12*c_d2dfx2*(two*f[-2:]-five*f[-3:-1]+four*f[-4:-2]-f[-5:-3])
            #df[:2] = c_d2dfx2*(45*f[:2]+61*f[4:6]+214*f[2:4]-156*f[3:5]-154*f[1:3]-10*f[5:7])
            #df[-2:] = c_d2dfx2*(45*f[-2:]+61*f[-6:-4]+214*f[-4:-2]-156*f[-5:-3]-154*f[-3:-1]-10*f[-7:-5])
            return df
    else:
        def dfdx(f):
            df = c_dfdx*(np.roll(f, 2, axis=0)-eight*np.roll(f, 1, axis=0)
                         +eight*np.roll(f, -1, axis=0)-np.roll(f, -2, axis=0))
            return df
        def d2fdx2(f):
            df = c_d2dfx2*(16*np.roll(f, 1, axis=0)+16*np.roll(f, -1, axis=0)
                           -30*f-np.roll(f, 2, axis=0)-np.roll(f, -2, axis=0))
            return df
    return dfdx, d2fdx2