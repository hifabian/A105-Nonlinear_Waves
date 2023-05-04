import numpy as np
import matplotlib.pyplot as plt

from precision import *
from rhs.aux import get_rhs
from numerics.ssm import ssm_integrator
from config import readConfig


parafile = 'files/block.ini'
irun = 0

if __name__ == '__main__':
    # Read configuration
    params, discret, funcs, plotting = readConfig(parafile)
    γρ, g, ν, μ = params
    x, Nx, dx, Nt, dt, periodic, filter, normalized = discret
    v0, h0, vr, hr = funcs
    labels, colors = plotting
    irun = 0  # Only first

    # Initial
    q0 = np.empty((Nx, 2), dtype=wp)
    q0[:,0] = v0(x)
    q0[:,1] = np.log(h0(x))

    # Solve ODE
    f = get_rhs(γρ[irun], ν[irun], g[irun], μ[irun], dx, Nx, periodic, normalized)
    t, q = ssm_integrator(f, q0, dt, dt*Nt, filter=filter, N=1)

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(4, 6))
    extent = [np.min(x), np.max(x), 0, t]
    axes[0].imshow(q[:,:,0], extent=extent, cmap='binary', origin='lower', aspect='auto')
    axes[1].imshow(np.exp(q[:,:,1]), extent=extent, cmap='binary', origin='lower', aspect='auto', vmax=2)
    tt = np.linspace(0, t, num=len(x))

    # Velocities at interfaces (first shock, second shock, rarefaction)
    v1 = 1.0; v2 = 0.5; v3 = 1.0; v4 = 1.0; v5 = 0.5
    # Height at second shock
    h3 = 2.0; h4 = 1.0
    if g != 0:
        axes[1].plot(x+0.4, ((v1+v2)-np.sqrt((v1+v2)**2-8*g*x))/(2*g), '--', c='red')
        axes[1].plot(x+0.4, (v3*h3-v4*h4-np.sqrt((v3*h3-v4*h4)**2-4*(h3-h4)**2*g*x))/((h3-h4)*g), '--', c='red')
        axes[1].plot(x+0.2, (v5-np.sqrt(v5**2-4*g*x))/g, '--', c='blue')
    else:
        axes[1].plot(x+0.4, 2/(v1+v2)*x, '--', c='red')
        axes[1].plot(x+0.4, 2*(h3-h4)/(v3*h3-v4*h4)*x, '--', c='red')
        axes[1].plot(x+0.2, 2/v5*x, '--', c='blue')

    axes[0].set_xlim(np.min(x), np.max(x))
    axes[0].set_ylim(0, t)
    #axes[0].set_xlabel('$z$')
    axes[0].set_xticks([])
    axes[0].set_ylabel('$t$')
    axes[0].set_xticks([np.min(x), (np.max(x)+np.min(x))/2, np.max(x)])
    axes[0].set_yticks([0, t/2, t])
    axes[1].set_xlim(np.min(x), np.max(x))
    axes[1].set_ylim(0, t)
    axes[1].set_xlabel('$z$')
    #axes[1].set_yticks([])
    axes[1].set_ylabel('$t$')
    axes[1].set_yticks([0, t/2, t])
    axes[1].set_xticks([np.min(x), (np.max(x)+np.min(x))/2, np.max(x)])
    plt.tight_layout()
    plt.savefig('plots/xt-'+parafile.split('/')[-1][:-4]+'.pdf')
    plt.show()