import numpy as np
from precision import *
from numerics.fd import *


def get_rhs(γρ, ν, g, μ, dx, Nx, periodic):
    res = np.empty((Nx, 2))
    dfdx, d2fdx2 = fourth_order(dx, periodic)

    def f(q, t):
        dq = dfdx(q)
        hm2 = q[:,1]**-2
        dq2 = dq[:,1]*dq[:,1]
        odf2 = one+four*q[:,1]*q[:,1]*dq2
        fddf = q[:,1]**d2fdx2(q[:,1])
        p = (hm2/np.sqrt(odf2)-two*(dq2+fddf)/odf2**threehalf)

        res[:,0] = -(q[:,0]-g*t)*dq[:,0] \
                   + three*ν*d2fdx2(q[:,0]) + three*ν*dq[:,0]*four*dq[:,1]/q[:,1] \
                   - γρ*dfdx(p)
        res[:,1] = -fourth*q[:,1]*dq[:,0] - (q[:,0]-g*t)*dq[:,1] + two*μ*(fddf+dq2)
        if not periodic: #-> fixed nozzle
            res[0,0] = 0
            res[0,1] = 0
            res[-1,0] = 0
            res[-1,1] = 0
        return res

    return f
