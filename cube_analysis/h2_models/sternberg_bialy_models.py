
'''
H2/HI models from Sternberg+14, Bialy+16,17
'''

import numpy as np
import astropy.units as u


def sternberg2014_model(Iuv, n, phi_g=1., Z=1.):
    '''
    Returns Sigma_HI from the Sternberg+2014 model.

    Parameters
    ----------
    Iuv : float or dimensionless
        FUV intensity relative to Draine field
    n : Quantity
        Volume density.
    phi_g : float
        Const. of order unity depending on dust-grain
        absorption properties.
    Z : float
        Metallicity relative to solar.
    '''

    alpG = 1.54 * Iuv * (100 * u.cm**-3 / n.to(u.cm**-3)) * \
        phi_g / (1 + (2.64 * phi_g * Z)**0.5)

    return (9.5 / (phi_g * Z)) * np.log((alpG / 3.2) + 1)


def sternberg2014_sigmaHI_H2(Sigma_gas, Iuv, n, phi_g=1., Z=1.):

    Sigma_HI = sternberg2014_model(Iuv, n, phi_g, Z)

    Sigma_H2 = Sigma_HI * (1 - (Sigma_HI / Sigma_gas))

    # Set Sigma_H2 = 0. when Sigma_HI < Sigma_gas
    Sigma_H2[Sigma_gas < Sigma_HI] = 0.

    return Sigma_HI, Sigma_H2


def sternberg2014_model_const_alphG(Z=1.):
    '''
    Returns Sigma_HI from the Sternberg+2014 model.

    See Eq. 5 from Schruba+2018

    Parameters
    ----------
    Z : float
        Metallicity relative to solar.
    '''

    return (5.6 / Z) * u.solMass / u.pc**2


def sternberg2014_model_const_alphG_avg(Z=1., fA=1., phi_diff=1.):
    '''
    Returns Sigma_HI from the Sternberg+2014 model.

    See Eq. 8 from Schruba+2018

    Valid when the beam averages over many clouds.

    Parameters
    ----------
    Z : float
        Metallicity relative to solar.
    fA : float
        Filling factor of GMCs within beam
    phi_diff : float
        Total mass in the diffuse HI
    '''

    return (5.6 / Z) * fA * (1 + phi_diff) * u.solMass / u.pc**2
