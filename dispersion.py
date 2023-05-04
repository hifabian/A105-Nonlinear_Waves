#!python3
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt


if __name__ == '__main__':
    k = np.linspace(-2, 2, 1000)
    ν = 0.5
    γρ = 5.0
    h0 = 1.0
    σ = lambda k: -1.5*ν*k**2+0.5*np.sqrt(γρ*k**2/h0*2*(1-h0**2*k**2)+9*k**4*ν**2, dtype=complex)
    σ2 = lambda k: -1.5*ν*k**2-0.5*np.sqrt(γρ*k**2/h0*2*(1-h0**2*k**2)+9*k**4*ν**2, dtype=complex)

    plt.figure(figsize=(6, 4))
    plt.plot(k*h0, np.real(σ2(k)), '-', c='gray')
    plt.plot(k*h0, np.imag(σ2(k)), '--', c='gray')
    plt.plot(k*h0, np.real(σ(k)), '-k', label='$\\mathcal{Real}~[\\sigma(k)]$')
    plt.plot(k*h0, np.imag(σ(k)), '--k', label='$\\mathcal{Imag}~[\\sigma(k)]$')
    plt.vlines([1./12*2*np.pi, 2./12*2*np.pi, 3./12*2*np.pi],
               ymin=-5, ymax=5,
               colors=['blue', 'green', 'red'], linestyles='dotted')
    plt.xlabel('$h_0 k$')
    plt.ylabel(f'$\\sigma$')
    #plt.title(f'Dispersion Relation for $\\nu = {ν}$, $\\frac{{\\gamma}}{{\\rho}} = {γρ}$ and $h_0 = {h0}$')
    plt.xlim(-0.1, 2)
    plt.ylim(-3.9, 3.9)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/dispersion.pdf')
    plt.show()
