import numpy as np
import matplotlib.pyplot as plt

def metrics(X, Y):  # return(MSE,eta,sig)
    # MSE calculation
    MSE = ((X - Y) ** 2).mean(axis=0)
    # Phot_z metrics
    delta_z = (X - Y) / (1 + X)
    ntot = len(delta_z)
    nout = 0.
    for i in range(0, ntot):
        if abs(delta_z[i]) > .15:
            nout = nout + 1
    eta = 100. * nout / ntot
    sig = 1.48 * np.median(abs(delta_z))
    return MSE, eta, sig

def sigmaNMAD(X, Y):  # return(MSE,eta,sig)
    delta_z = (X - Y) / (1 + X)
    sig = 1.48 * np.median(abs(delta_z))
    return sig

def astro_plot(X, Y, title):
    x2i = np.arange(0., max(X), 0.1)
    line2 = x2i
    sigup1 = x2i + .05 * (1 + x2i)
    sigup2 = x2i + .15 * (1 + x2i)
    sigdown1 = x2i - .05 * (1 + x2i)
    sigdown2 = x2i - .15 * (1 + x2i)

    (MSE, eta, sig) = metrics(X, Y)
    plt.scatter(X, Y, marker='.', s=20)
    plt.plot(x2i, line2, 'r-', x2i, sigup1, 'r--', x2i, sigdown1, 'r--', x2i, sigup2, 'r:', x2i, sigdown2, 'r:',
             linewidth=1.5)
    plt.xlabel('$z_{spec}$', fontsize=14)
    plt.ylabel('$z_{phot}$', fontsize=14)
    plt.title(title)
    plt.annotate('$N_{tot} =' + str(len(X)) + '$', xy=(.01, .9), xycoords='axes fraction', xytext=(0.03, 0.8),
                 textcoords='axes fraction', fontsize=14)
    plt.annotate('$\eta =' + str(round(eta, 2)) + '\% $', xy=(.01, .2), xycoords='axes fraction', xytext=(0.03, 0.7),
                 textcoords='axes fraction', fontsize=14)
    plt.annotate('$\sigma_{NMAD} =' + str(round(sig, 3)) + '$', xy=(.01, .2), xycoords='axes fraction',
                 xytext=(0.03, 0.6), textcoords='axes fraction', fontsize=14)

