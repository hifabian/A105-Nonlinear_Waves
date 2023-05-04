import numpy as np
from precision import *
from numerics.fd import *


def get_rhs(γρ, ν, g, μ, dx, Nx, periodic, normalized):
    res = np.empty((Nx, 2))
    dfdx, d2fdx2 = second_order(dx, periodic)

    def f(q, t):
        dq = dfdx(q)
        if normalized:
            res[:,0] = -(q[:,0]-g*t)*dq[:,0] \
                         + three*ν*d2fdx2(q[:,0])
            res[:,1] = -half*dq[:,0] - half*(q[:,0]-g*t)*dq[:,1] + μ*(d2fdx2(q[:,1])+dq[:,1]*dq[:,1])
            if not periodic:
                if q[0,0]-g*t > 0:
                    res[0,0] = zero
                    res[0,1] = zero
                if q[-1,0]-g*t < 0:
                    res[-1,0] = zero
                    res[-1,1] = zero
        else:
            res[:,0] = -q[:,0]*dq[:,0] - g \
                         + three*ν*d2fdx2(q[:,0])
            res[:,1] = -half*dq[:,0] - half*q[:,0]*dq[:,1] + μ*(d2fdx2(q[:,1])+dq[:,1]*dq[:,1])
            if not periodic:
                if q[0,0] > 0:
                    res[0,0] = -g
                    res[0,1] = zero
                if q[-1,0] < 0:
                    res[-1,0] = -g
                    res[-1,1] = zero
        return res

    return f
