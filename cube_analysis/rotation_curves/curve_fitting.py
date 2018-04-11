
import numpy as np
from scipy.optimize import curve_fit
from astropy.table import Table
import astropy.units as u
import astropy.wcs as wcs
from galaxies import Galaxy


def vcirc_brandt(r, *pars):
    '''
    Fit Eq. 5 from Meidt+08 (Eq. 1 Faber & Gallagher 79)
    '''

    n, vmax, rmax = pars
    numer = vmax * (r / rmax)
    denom = np.power((1 / 3.) + (2 / 3.) *
                     np.power(r / rmax, n), (3 / (2 * n)))
    return numer / denom


def dvdR_brandt(r, *pars):
    '''
    Analytical derivative of the Brandt curve.
    '''
    n, vmax, rmax = pars

    r_rat = r / rmax

    term1 = (vmax * r_rat) * r_rat**(n - 1) / \
        np.power((1 / 3.) + (2 / 3.) * (r_rat**n), (3 / (2 * n)) + 1)

    term2 = vcirc_brandt(r, *pars) / r

    return - term1 + term2


def oortA_brandt(r, *pars):
    '''
    Oort A coefficient for local shear.
    '''

    return 0.5 * ((vcirc_brandt(r, *pars) / r) - dvdR_brandt(r, *pars))


def oortB_brandt(r, *pars):
    '''
    Oort B coefficient
    '''

    return - 0.5 * ((vcirc_brandt(r, *pars) / r) + dvdR_brandt(r, *pars))


def epifreq_brandt(r, *pars):
    '''
    Epicyclic frequency based on Brandt curve.
    '''

    Omega = (vcirc_brandt(r, *pars) / r)

    return np.sqrt(2 * Omega * (Omega - dvdR_brandt(r, *pars)))


def generate_vrot_model(table_name, model=vcirc_brandt, verbose=False):
    '''
    Parameters
    ----------
    table_name : str
        Name and path of the csv table produced by `run_diskfit.py`
    '''

    if isinstance(table_name, basestring):
        data = Table.read(table_name)
    else:
        data = table_name

    pars, pcov = curve_fit(model, data['r'], data['Vt'], sigma=data['eVt'],
                           absolute_sigma=True, p0=(1., 100., 1000.))

    if verbose:
        print("n: {0} +/- {1}".format(pars[0], np.sqrt(pcov[0, 0])))
        print("vmax: {0} +/- {1}".format(pars[1], np.sqrt(pcov[1, 1])))
        print("rmax: {0} +/- {1}".format(pars[2], np.sqrt(pcov[2, 2])))

    return pars, pcov


def return_smooth_model(table_name, header, gal, model=vcirc_brandt):

    assert isinstance(gal, Galaxy)

    radii = gal.radius(header=header).value
    pas = gal.position_angles(header=header).value

    scale = wcs.utils.proj_plane_pixel_scales(wcs.WCS(header))[0]
    # Distance scaling (1" ~ 4 pc). Conversion is deg to kpc
    dist_scale = (np.pi / 180.) * gal.distance.to(u.pc).value

    pars, pcov = generate_vrot_model(table_name, model=model, verbose=False)

    # Convert rmax to pc
    mod_pars = pars.copy()
    mod_pars[2] *= scale * dist_scale

    # Put into m/s.
    smooth_model = (model(radii, *mod_pars) * 1000.) * np.cos(pas) * \
        np.sin(gal.inclination).value

    # Shift by Vsys (m / s)
    smooth_model += gal.vsys.to(u.m / u.s).value

    return smooth_model
