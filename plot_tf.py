import numpy as np
import matplotlib.pyplot as plt

from precision import *
from rhs.log import get_rhs
from numerics.ssm import ssm_integrator
from config import readConfig


parafile = 'files/gravity.ini'

if __name__ == '__main__':
    # Read configuration
    params, discret, funcs, plotting = readConfig(parafile)
    γρ, g, ν, μ = params
    x, Nx, dx, Nt, dt, periodic, filter, normalized = discret
    v0, h0, vr, hr = funcs
    labels, colors = plotting
    noruns = len(γρ)

    # Initial
    q0 = np.empty((Nx, 2), dtype=wp)
    q0[:,0] = v0(x)
    q0[:,1] = np.log(h0(x))
    t = zero

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(5, 4))
    ## Initial
    axes[0].plot(x, q0[:,0], '--k', label='$v_0$')
    axes[1].plot(x, np.exp(q0[:,1]), '--k', label='$h_0$')

    ## Run all parameters
    for irun in range(noruns):
        lstr = ''
        if labels is not None:
            lstr = ', '
            lstr += labels[0]+'='+''.join(str(eval(labels[0])[irun]))
        # Integration
        f = get_rhs(γρ[irun], ν[irun], g[irun], μ[irun], dx, Nx, periodic, normalized)
        t, q = ssm_integrator(f, q0, dt, dt*Nt, filter=filter)
        axes[0].plot(x, q[:,0], c=colors[irun], label='$v_f$'+lstr)
        axes[1].plot(x, np.exp(q[:,1]), c=colors[irun], label='$h_f$'+lstr)

    ## Reference
    if vr is not None:
        axes[0].plot(x, vr(t, x), ':k', label='Reference')
    if hr is not None:
        axes[1].plot(x, hr(t, x), ':k', label='Reference')

    if g == 0 or not normalized:
        axes[0].set_ylabel('$v(z)$')
    else:
        axes[0].set_ylabel('$v(z)-gt$')
    #axes[0].legend()
    axes[1].set_xlabel('$z$')
    axes[1].set_ylabel('$h(z)$')
    #axes[1].legend()
    plt.tight_layout()
    plt.savefig('plots/tf-'+parafile.split('/')[-1][:-4]+'.pdf')
    plt.show()