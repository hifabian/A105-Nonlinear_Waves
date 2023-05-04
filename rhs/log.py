import numpy as np
from precision import *
from numerics.fd import *


def get_rhs(γρ, ν, g, μ, dx, Nx, periodic, normalized):
    res = np.empty((Nx, 2))
    dfdx, d2fdx2 = second_order(dx, periodic)

    def f(q, t):
        dq = dfdx(q)
        d2fdv = dfdx(np.exp(2*q[:,1])*dq[:,0])
        hm2 = np.exp(-2*q[:,1])
        odf2 = hm2+dq[:,1]*dq[:,1]
        p = (one/np.sqrt(odf2)-(dq[:,1]*dq[:,1]+d2fdx2(q[:,1]))/odf2**threehalf)

        if normalized:
            res[:,0] = -(q[:,0]-g*t)*dq[:,0] + two*γρ*p*hm2*dq[:,1]\
                         + three*ν*d2fdv*hm2 - γρ*hm2*dfdx(p)
            res[:,1] = -half*dq[:,0] - (q[:,0]-g*t)*dq[:,1] + μ*(d2fdx2(q[:,1])+dq[:,1]*dq[:,1])
            if not periodic:
                if q[0,0]-g*t > 0:
                    res[0,0] = zero
                    res[0,1] = zero
                if q[-1,0]-g*t < 0:
                    res[-1,0] = zero
                    res[-1,1] = zero
                pass
        else:
            res[:,0] = -q[:,0]*dq[:,0] + two*γρ*p*hm2*dq[:,1] - g \
                         + three*ν*d2fdv*hm2 - γρ*hm2*dfdx(p)
            res[:,1] = -half*dq[:,0] - q[:,0]*dq[:,1] + μ*(d2fdx2(q[:,1])+dq[:,1]*dq[:,1])
            if not periodic: #-> fixed nozzle
                if q[0,0] > 0:
                    res[0,0] = 0.01*one/(one+np.exp(-0.2*(t-10)))*np.cos(1*np.pi*t)
                    #res[0,0] = zero
                    res[0,1] = zero
                if q[-1,0] < 0:
                    res[-1,0] = -g
                    res[-1,1] = zero
                pass
        return res

    return f
