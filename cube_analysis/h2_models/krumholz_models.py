
import numpy as np
from itertools import cycle
import scipy.optimize as opt
import astropy.units as u

'''
Krumholz model for the fraction of H2 as a function of surface density and
metallicity.

Following Schruba+11, the output is in terms of the H2-HI fraction rather than
the the molecular fraction.
'''


def krumholz2009a_ratio_model(Sigma, Z=0.5, c=1):
    '''

    Eq. 2 from Krumholz et al., 2009, 699, 850-856.

    Parameters
    ----------
    sigma : float or np.ndarray
        Surface Density in Msol/pc^2
    Z : float
        Metallicity in units of solar metallicity.
    c : float
        Clumping fraction. Expected to be near unity with a resolution of
        100 pc. c>=1.

    Returns
    -------
    RH2 : float or np.ndarray
        The ratio of H2 to HI.
    '''

    Sigma_comp = c * Sigma

    chi = 0.77 * (1 + 3.1 * np.power(Z, 0.365))

    s = np.log(1 + 0.6 * chi) / (0.04 * Sigma_comp * Z)

    delta = 0.0712 * np.power((0.1 / s) + 0.675, -2.8)

    frac = 1 - np.power(1 + np.power(0.75 * (s / (1 + delta)), -5.), -0.2)

    return frac / (1 - frac)


def krumholz2009b_ratio_model(Sigma, psi=1.0, c=1, Z=0.1):
    '''

    Eq. 38 & 39 from Krumholz et al., 2009, 693, 216-235.

    Parameters
    ----------
    sigma : float or np.ndarray
        Surface Density in Msol/pc^2
    psi : float
        Dust-adjusted radiation field (unitless). Related to metallicity
        (among other things). At Z=1, psi=1.6, and at Z=0.1, psi=1.0
    c : float
        Clumping fraction. Expected to be near unity with a resolution of
        100 pc. c>=1.

    Returns
    -------
    RH2 : float or np.ndarray
        The ratio of H2 to HI.
    '''

    Sigma_comp = c * Sigma

    s = Sigma_comp * Z / float(psi)

    term1 = (s / 11.) ** 3 * ((125 + s) / (96 + s)) ** 3

    return np.power(1 + term1, 1 / 3.) - 1


def krumholz2013_ratio_model(Sigma, Z=0.5, c=1):
    '''
    Eq. 10-13 from Krumholz 2013, MNRAS, 436, 2747.

    Checked against this implementation (07/2017):
    https://bitbucket.org/krumholz/kbfc17

    Parameters
    ----------
    sigma : float or np.ndarray
        Surface Density in Msol/pc^2
    Z : float
        Metallicity in units of solar metallicity.
    c : float
        Clumping fraction. Expected to be near unity with a resolution of
        100 pc. c>=1.

    Returns
    -------
    RH2 : float or np.ndarray
        The ratio of H2 to HI.
    '''

    chi = 3.1 * (1 + 3.1 * Z**0.365) / 4.1

    tauc = 0.066 * (c * Sigma) * Z

    s = np.log(1 + 0.6 * chi + 0.01 * chi**2) / (0.6 * tauc)

    frac = np.maximum(1.0 - 0.75 * s / (1.0 + 0.25 * s), 1e-10)

    return frac / (1 - frac)


def krumholz2013_sigmaHI(Sigma, Z=0.5, c=1):
    '''
    Return the predicted HI sigma as a function of the total Sigma.
    '''

    RH2 = krumholz2013_ratio_model(Sigma, Z=Z, c=c)

    SigmaHI = Sigma / (1 + RH2)

    return SigmaHI


def krumholz2013_sigmaHI_H2(Sigma, Z=0.5, c=1):
    '''
    Return the predicted HI and H2 sigma as a function of the total Sigma.
    '''

    RH2 = krumholz2013_ratio_model(Sigma, Z=Z, c=c)

    SigmaHI = Sigma / (1 + RH2)

    SigmaH2 = RH2 * SigmaHI

    return SigmaHI, SigmaH2


def optimize_clump_factors(Sigma, R, Z=0.5, c_init=1.):
    '''
    Solve for the clump factor needed to intersect each point.

    Parameters
    ----------
    Sigma : Quantity
        Surface densities.
    R : array
        Surface density ratios.
    Z : float or array
        Metallicity values.
    '''

    if not isinstance(Z, np.ndarray):
        Z = cycle([Z])

    clump_values = []

    for sig, r, z in zip(Sigma, R, Z):

        def model(sigma, c):
            return krumholz2013_ratio_model(sigma, Z=z, c=c)

        popt, pcov = opt.curve_fit(model, sig, r, p0=(c_init))

        clump_values.append(popt[0])

    return np.array(clump_values)


def krumholz_maxhi_sigma(c=1.0, Z=1.0):
    '''
    Return the maximum HI surface density.

    Eq. 26 from Krumholz, 2013, MNRAS, 436, 2747-2762.

    Parameters
    ----------
    c : float, optional
        Clumping factor.
    Z : float, optional.
        Metallicity.

    Returns
    -------
    Sigma_HI_max : float,
        HI surface density in Msol/pc^2.
    '''

    chi = (3.1 / 4.1) * (1 + 3.1 * Z**0.365)
    term1 = np.log(1 + 0.6 * chi + 0.01 * chi**2)

    return (24. / (c * Z)) * (term1 / 1.29) * u.solMass / u.pc**2
