import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from precision import *
from rhs.aux import get_rhs
from numerics.ssm import rk4
from config import readConfig


parafile = 'files/block.ini'

if __name__ == '__main__':
    Nplot = 100
    # Read configuration
    params, discret, funcs, plotting = readConfig(parafile)
    γρ, g, ν, μ = params
    x, Nx, dx, Nt, dt, periodic, filter, normalized = discret
    v0, h0, vr, hr = funcs
    labels, colors = plotting
    noruns = len(γρ)

    flt_coef = [0.0625, -0.25, 0.375, -0.25, 0.0625]
    Aflt     = cst(0.1)

    # Initial
    q0 = np.empty((Nx, 2), dtype=wp)
    q0[:,0] = v0(x)
    q0[:,1] = np.log(h0(x))
    t = zero; it = 0
    T = Nt*dt

    # ODE parameters
    fa = [get_rhs(γρ[irun], ν[irun], g[irun], μ[irun], dx, Nx, periodic, normalized) for irun in range(noruns)]
    ya = [np.copy(q0) for _ in range(noruns)]

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(5, 4))
    ims = []
    ## Initial
    line1, = axes[0].plot(x, ya[0][:,0], '--k', label='$v_0$')
    line2, = axes[1].plot(x, np.exp(ya[0][:,1]), '--k', label='$h_0$')
    ## Labels
    if g == 0 or not normalized:
        axes[0].set_ylabel('$v(z)$')
    else:
        axes[0].set_ylabel('$v(z)-gt$')
    axes[0].legend()
    axes[1].set_xlabel('$z$')
    axes[1].set_ylabel('$h(z)$')
    axes[1].legend()

    # Time loop
    while t < T:
        for irun in range(noruns):
            ya[irun] = rk4(fa[irun], ya[irun], dt, t)
            if filter:
                for i in range(2):
                    ya[irun][2:-2,i] -= Aflt*np.convolve(ya[irun][:,i], flt_coef,'same')[2:-2]
        t += dt; it += 1
        if it % Nplot == 0:
            print(it, t, T)
            lines1 = []; lines2 = []
            for irun in range(noruns):
                line1, = axes[0].plot(x,ya[irun][:,0], c=colors[irun], lw=1, ms=5, zorder=2)
                line2, = axes[1].plot(x,np.exp(ya[irun][:,1]), c=colors[irun], lw=1, ms=5, zorder=2)
                lines1.append(line1); lines2.append(line2)
            if vr is not None:
                rline, = axes[0].plot(x, vr(t, x), ':k')
                lines1.append(rline)
            if hr is not None:
                rline, = axes[1].plot(x, hr(t, x), ':k')
                lines2.append(rline)
            ims.append([*lines1, *lines2])

    # -- Create animation and display
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False, repeat_delay=2000)
    plt.tight_layout()
    ani.save('plots/tm-'+parafile.split('/')[-1][:-4]+'.mp4', fps=32)
    plt.show()