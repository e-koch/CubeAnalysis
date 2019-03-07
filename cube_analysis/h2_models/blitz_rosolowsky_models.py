
'''
Pressure models from Blitz & Rosolowsky 2004, 2006
'''

import astropy.units as u
import astropy.constants as con
import numpy as np


def Pext_star(Sigma_g, Sigma_star, sigma_g, h_star):
    '''
    Eq. 5 from BR06. Returns Pext / k_B
    '''

    return 272 * u.cm**-3 * u.K * Sigma_g.to(u.solMass / u.pc**2).value * \
        Sigma_star.to(u.solMass / u.pc**2).value**0.5 * \
        sigma_g.to(u.km / u.s).value * \
        h_star.to(u.pc).value**-0.5


def Pext_sum(Sigma_gmc, Sigma_HI, Sigma_star, H_mol=10 * u.pc,
             H_star=300 * u.pc, sigma_HI=7 * u.km / u.s):
    '''
    Multiple pressure terms on a GMC from Sun+in prep

    Three terms:
    - GMC pressure
    - Stellar pressure on GMC
    - HI pressure on GMC

    '''

    # Prefactor set by cloud shape. Assume sphere
    f = 3 / 8.

    P_GMC = f * np.pi * (con.G * Sigma_gmc**2).to(u.dyne / u.cm**2)

    rho_star = Sigma_star / H_star

    P_staronGMC = f * np.pi * \
        (con.G * rho_star * H_mol * Sigma_gmc).to(u.dyne / u.cm**2)

    Sigma_T = Sigma_gmc + Sigma_HI

    P_HIonGMC = Sigma_HI * (0.5 * np.pi * con.G * Sigma_T +
                            (2 * con.G * rho_star)**0.5 * sigma_HI)
    P_HIonGMC = P_HIonGMC.to(u.dyne / u.cm**2)

    P_T = P_GMC + P_staronGMC + P_HIonGMC

    # Convert to P / k_B, which is typically used in to be propto nT

    P_T = (P_T / con.k_B).to(u.cm**-3 * u.K)
    P_GMC = (P_GMC / con.k_B).to(u.cm**-3 * u.K)
    P_staronGMC = (P_staronGMC / con.k_B).to(u.cm**-3 * u.K)
    P_HIonGMC = (P_HIonGMC / con.k_B).to(u.cm**-3 * u.K)

    return P_T, P_GMC, P_staronGMC, P_HIonGMC
