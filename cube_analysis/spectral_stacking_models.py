
'''
Basic models for decomposing large-scale stacked profiles
'''

import numpy as np
from astropy.modeling import models, fitting
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import erf


def fit_2gaussian(vels, spectrum):
    '''
    Fit a 2 Gaussian model with the means tied.
    '''

    max_vel = vels[np.argmax(spectrum)]
    specmax = spectrum.max()

    # Estimate the inner width from the HWHM
    sigma_est = find_hwhm(vels, spectrum)[0]
    # Use this as the narrow estimate, and let the wide guess be 3x

    g_HI_init = models.Gaussian1D(amplitude=0.75 * specmax, mean=max_vel,
                                  stddev=sigma_est) +  \
        models.Gaussian1D(amplitude=0.25 * specmax, mean=max_vel,
                          stddev=3 * sigma_est)

    # Force to the same mean
    def tie_mean(mod):
        return mod.mean_0

    g_HI_init.mean_1.tied = tie_mean

    fit_g = fitting.LevMarLSQFitter()

    g_HI = fit_g(g_HI_init, vels, spectrum)

    # The covariance matrix is hidden away... tricksy
    cov = fit_g.fit_info['param_cov']
    parnames = [n for n in g_HI.param_names if n not in ['mean_1']]
    parvals = [v for (n, v) in zip(g_HI.param_names, g_HI.parameters)
               if n in parnames]

    if cov is not None:
        chan_width = np.diff(vels[:2])[0]
        parerrs = []
        for name, var in zip(parnames, np.diag(cov)):
            if "mean" in name or "stddev" in name:
                # Add the finite channel width in quadrature
                stderr = np.sqrt(var + (0.5 * chan_width)**2)
            else:
                stderr = np.sqrt(var)

            parerrs.append(stderr)
    else:
        parerrs = [np.NaN] * len(parnames)

    return parvals, parerrs, cov, parnames, g_HI


def fit_gaussian(vels, spectrum, p0=None):
    '''
    Fit a Gaussian model.
    '''

    if p0 is None:
        max_vel = vels[np.argmax(spectrum)]
        specmax = spectrum.max()

        # Estimate the inner width from the HWHM
        sigma_est = find_hwhm(vels, spectrum)[0]
    else:
        specmax, max_vel, sigma_est = p0

    # Use this as the narrow estimate, and let the wide guess be 3x

    g_HI_init = models.Gaussian1D(amplitude=specmax, mean=max_vel,
                                  stddev=sigma_est)

    fit_g = fitting.LevMarLSQFitter()

    g_HI = fit_g(g_HI_init, vels, spectrum)

    # The covariance matrix is hidden away... tricksy
    cov = fit_g.fit_info['param_cov']
    if cov is None:
        cov = np.zeros((3, 3)) * np.NaN
    parnames = g_HI.param_names
    parvals = g_HI.parameters

    # Sometimes the width is negative
    parvals[-1] = np.abs(parvals[-1])

    if cov is not None:
        chan_width = np.diff(vels[:2])[0]
        parerrs = []
        for name, var in zip(parnames, np.diag(cov)):
            if "mean" in name or "stddev" in name:
                # Add the finite channel width in quadrature
                stderr = np.sqrt(var + (0.5 * chan_width)**2)
            else:
                stderr = np.sqrt(var)

            parerrs.append(stderr)
    else:
        parerrs = [np.NaN] * len(parnames)

    return parvals, parerrs, cov, parnames, g_HI


def find_hwhm(vels, spectrum):
    '''
    Return the equivalent Gaussian sigma based on the HWHM positions.
    '''

    halfmax = spectrum.max() * 0.5

    # Model the spectrum with a spline
    # x values must be increasing for the spline, so flip if needed.
    spec_for_interp = spectrum if np.diff(vels[:2])[0] > 0 else spectrum[::-1]
    vels_for_interp = vels if np.diff(vels[:2])[0] > 0 else vels[::-1]

    interp1 = InterpolatedUnivariateSpline(vels_for_interp,
                                           spec_for_interp - halfmax, k=3)

    fwhm_points = interp1.roots()
    if len(fwhm_points) < 2:
        raise ValueError("Found less than 2 roots!")
    # Only keep the min/max if there are multiple
    fwhm_points = (min(fwhm_points), max(fwhm_points))

    fwhm = fwhm_points[1] - fwhm_points[0]

    # Convert to equivalent Gaussian sigma
    sigma = fwhm / np.sqrt(8 * np.log(2))

    return sigma, fwhm_points, vels_for_interp, spec_for_interp


def fit_hwhm(vels, spectrum, asymm='full'):
    '''
    Scale the inner Gaussian to the HWHM of the profile.

    Extracts the equivalent inner gaussian width, the fraction of flux in the
    wings, the equivalent width of the wings, and the asymmetry of the wings or
    profile. Each of these is defined in Stilp et al. (2013).
    https://ui.adsabs.harvard.edu/#abs/2013ApJ...765..136S/abstract

    One additional parameter has been added:

    kappa = Sum_i (Data_i - Gauss_i) / Sum_i (Gauss_i)
    for i within the FWHM.

    kappa measures the kurtic behaviour of the peak. If negative, the peak
    is leptokurtic (more peaked). If positive, the peak is playkurtic
    (less peaked).

    Parameters
    ----------
    asymm : str, optional
        'full' to calculate asymmetry of whole profile, or 'wings' to
        only calculate it in the wings.
    '''

    sigma, fwhm_points, vels_for_interp, spec_for_interp = \
        find_hwhm(vels, spectrum)

    maxpos = np.argmax(spec_for_interp)
    maxvel = vels_for_interp[maxpos]
    maxval = spectrum.max()

    # Define a Gaussian with this width
    hwhm_gauss = models.Gaussian1D(amplitude=maxval, mean=maxvel,
                                   stddev=sigma)

    low_mask = vels_for_interp < fwhm_points[0]
    high_mask = vels_for_interp > fwhm_points[1]

    tail_flux_excess = \
        (np.sum([spec - hwhm_gauss(vel) for spec, vel in
                 zip(spec_for_interp[low_mask], vels_for_interp[low_mask])]) +
         np.sum([spec - hwhm_gauss(vel) for spec, vel in
                 zip(spec_for_interp[high_mask], vels_for_interp[high_mask])]))

    # Calculate fraction in the wings
    f_wings = tail_flux_excess / np.sum(spectrum)

    # Equivalent sigma^2 of the wings
    var_wing = (np.sum([vel**2 * (spec - hwhm_gauss(vel)) for spec, vel in
                        zip(spec_for_interp[low_mask],
                            vels_for_interp[low_mask])]) +
                np.sum([vel**2 * (spec - hwhm_gauss(vel)) for spec, vel in
                        zip(spec_for_interp[high_mask],
                            vels_for_interp[high_mask])])) /\
        tail_flux_excess
    sigma_wing = np.sqrt(var_wing)

    if asymm == "full":
        neg_iter = range(maxpos - 1, -1, -1)
        pos_iter = range(maxpos + 1, len(vels))
        asymm_val = np.sum([np.sqrt((spec_for_interp[vel_l] -
                                     spec_for_interp[vel_u])**2)
                            for vel_l, vel_u in zip(neg_iter, pos_iter)]) / \
            np.sum(spectrum)
    elif asymm == 'wings':
        neg_iter = np.where(vels < fwhm_points[0])[0]
        pos_iter = np.where(vels > fwhm_points[1])[0]
        asymm_val = np.sum([np.sqrt((spec_for_interp[vel_l] -
                                     spec_for_interp[vel_u])**2)
                            for vel_l, vel_u in zip(neg_iter, pos_iter)]) / \
            tail_flux_excess
    else:
        raise TypeError("asymm must be 'full' or 'wings'.")

    fwhm_mask = np.logical_and(vels_for_interp > fwhm_points[0],
                               vels_for_interp < fwhm_points[1])

    # Fraction of Gaussian area within the FWHM
    fwhm_area_conv = erf(np.sqrt(np.log(2)))
    fwhm_area = maxval * np.sqrt(2 * np.pi) * sigma * fwhm_area_conv

    diff_vel = np.diff(vels_for_interp[:2])[0]

    kappa = np.sum([(spec - hwhm_gauss(vel)) for spec, vel in
                    zip(spec_for_interp[fwhm_mask],
                        vels_for_interp[fwhm_mask])]) * diff_vel / fwhm_area

    params = np.array([sigma, f_wings, sigma_wing, asymm_val, kappa])
    param_names = ["sigma,", "f_wings", "sigma_wing", "asymm", "kappa"]

    return params, param_names, hwhm_gauss
