
'''
Calculate radial mass flow from the radial velocity component
'''

import numpy as np
import astropy.units as u


def rad_flow(surfdens, circ_tab, gal, pix_to_phys):
    '''
    See Eq'n 10 in Schmidt+2016
    '''

    radii = gal.radius(header=surfdens.header).to(u.kpc)

    rad_bins = (circ_tab['r'] * u.pix * pix_to_phys).to(u.kpc)
    bin_width = np.diff(rad_bins[:2])[0]

    in_rad = rad_bins - bin_width / 2.
    out_rad = rad_bins + bin_width / 2.

    vrad = circ_tab['Vr'] * u.km / u.s
    vrad_err = circ_tab['eVr'] * u.km / u.s

    mass_flux = np.zeros_like(vrad.value)
    mass_flux_err = np.zeros_like(vrad.value)

    for i, (in_r, out_r, vr, vre) in enumerate(zip(in_rad, out_rad, vrad,
                                                   vrad_err)):

        mask = np.logical_and(radii >= in_r, radii < out_r)

        mass_flux_r = vr * np.nansum(surfdens[mask]) * pix_to_phys * u.pix
        mass_flux_err_r = vre * np.nansum(surfdens[mask]) * pix_to_phys * u.pix

        mass_flux[i] = mass_flux_r.to(u.solMass / u.yr).value
        mass_flux_err[i] = mass_flux_err_r.to(u.solMass / u.yr).value

    mass_flux = mass_flux * u.solMass / u.yr
    mass_flux_err = mass_flux_err * u.solMass / u.yr

    return mass_flux, mass_flux_err


def vr_map(header, circ_tab, gal, pix_to_phys):
    '''
    Generate a map matching the given header that gives the 2D distribution
    of radial velocities.
    '''

    radii = gal.radius(header=header).to(u.kpc)

    rad_bins = (circ_tab['r'] * u.pix * pix_to_phys).to(u.kpc)
    bin_width = np.diff(rad_bins[:2])[0]

    in_rad = rad_bins - bin_width / 2.
    out_rad = rad_bins + bin_width / 2.

    vrad = circ_tab['Vr'] * u.km / u.s
    vrad_err = circ_tab['eVr'] * u.km / u.s

    vrad_map = np.zeros_like(radii.value) * u.km / u.s
    vrad_map_err = np.zeros_like(radii.value) * u.km / u.s

    for i, (in_r, out_r, vr, vre) in enumerate(zip(in_rad, out_rad, vrad,
                                                   vrad_err)):

        mask = np.logical_and(radii >= in_r, radii < out_r)

        vrad_map[mask] = vrad.value
        vrad_map_err[mask] = vrad_err.value

    return vrad_map, vrad_map_err
