import numpy as np
from precision import *
from numerics.fd import *


def get_rhs(γρ, ν, g, μ, dx, Nx, periodic):
    res = np.empty((Nx, 2))
    dfdx, d2fdx2 = fourth_order(dx, periodic)

    def f(q):
        dq = dfdx(q)
        h2 = q[:,1]*q[:,1]
        dh2dv = dfdx(h2*dq[:,0])
        ddh = d2fdx2(q[:,1])
        dh2o = one+dq[:,1]*dq[:,1]
        p = γρ*(one/(q[:,1]*(dh2o)**half)-ddh/(dh2o)**threehalf)

        res[:,0] = -q[:,0]*dq[:,0] - g \
                   + three*ν*dh2dv / h2- dfdx(p)
        res[:,1] = -half*q[:,1]*dq[:,0] - q[:,0]*dq[:,1] +  μ*(d2fdx2(q[:,1]))
        if not periodic: #-> fixed nozzle
            res[0,0] = 0
            res[0,1] = 0
            res[-1,0] = 0
            res[-1,1] = 0
        return res

    return f