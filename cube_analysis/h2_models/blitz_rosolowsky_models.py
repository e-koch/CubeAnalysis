
'''
Pressure models from Blitz & Rosolowsky 2004, 2006
'''

import astropy.units as u


def Pext_star(Sigma_g, Sigma_star, sigma_g, h_star):
    '''
    Eq. 5 from BR06. Returns Pext / k_B
    '''

    return 272 * u.cm**-3 * u.K * Sigma_g.to(u.solMass / u.pc**2).value * \
        Sigma_star.to(u.solMass / u.pc**2).value**0.5 * \
        sigma_g.to(u.km / u.s).value * \
        h_star.to(u.pc).value**-0.5
