
'''
Basic models for decomposing large-scale stacked profiles
'''

import numpy as np
from astropy.modeling import models, fitting
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import erf
from functools import partial


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


def reorder_spectra(vels, spectrum):
    spec_for_interp = spectrum if np.diff(vels[:2])[0] > 0 else spectrum[::-1]
    vels_for_interp = vels if np.diff(vels[:2])[0] > 0 else vels[::-1]

    return spec_for_interp, vels_for_interp


def find_hwhm(vels, spectrum, interp_factor=10):
    '''
    Return the equivalent Gaussian sigma based on the HWHM positions.
    '''

    halfmax = spectrum.max() * 0.5

    # Model the spectrum with a spline
    # x values must be increasing for the spline, so flip if needed.
    spec_for_interp, vels_for_interp = reorder_spectra(vels, spectrum)

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

    # Upsample in velocity to estimate the peak position
    interp_factor = float(interp_factor)
    chan_size = np.diff(vels_for_interp[:2])[0] / interp_factor
    upsamp_vels = np.linspace(vels_for_interp.min(),
                              vels_for_interp.max() + 0.9 * chan_size,
                              vels_for_interp.size * interp_factor)
    upsamp_spec = interp1(upsamp_vels)
    peak_velocity = upsamp_vels[np.argmax(upsamp_spec)]

    return sigma, fwhm_points, peak_velocity


def _hwhm_fitter(vels, spectrum, hwhm_gauss, asymm='full', sigma_noise=None,
                 nbeams=1, interp_factor=10):

    spec_for_interp, vels_for_interp = reorder_spectra(vels, spectrum)

    sigma = hwhm_gauss.stddev.value
    peak_velocity = hwhm_gauss.mean.value
    maxval = hwhm_gauss.amplitude.value

    maxpos = np.abs(vels_for_interp - peak_velocity).argmin()

    hwhm_factor = np.sqrt(2 * np.log(2))
    fwhm_points = [hwhm_gauss.mean - hwhm_gauss.stddev * hwhm_factor,
                   hwhm_gauss.mean + hwhm_gauss.stddev * hwhm_factor]

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
                            vels_for_interp[high_mask])]))

    sigma_wing = np.sqrt(np.abs(var_wing / tail_flux_excess))

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

    # Estimate uncertainties if sigma_noise is given.
    param_stderrs = np.zeros((5,))
    if sigma_noise is not None:

        # Error for each value in the profile
        delta_S = sigma_noise * np.sqrt(nbeams)

        # Error in sigma is channel_width / 2 sqrt(2 ln 2), from the uncertainty
        # on each location at the FWHM having the channel width as an error
        chan_width = np.abs(np.diff(vels[:2])[0])

        delta_sigma = chan_width / (2 * np.sqrt(2 * np.log(2)))

        # Error in peak velocity should approach channel width ASSUMING the
        # shuffling is optimized
        delta_v_peak = chan_width / 2.

        g_uncert = partial(gauss_uncert, model=hwhm_gauss,
                           chan_width=chan_width,
                           nbeams=nbeams, sigma_noise=sigma_noise)

        # Assume the only significant error comes from the noise in the profile
        wing_term1 = (np.sum([delta_S + g_uncert(vel) for vel in
                              vels_for_interp[low_mask]]) +
                      np.sum([delta_S + g_uncert(vel) for vel in
                              vels_for_interp[high_mask]]))

        wing_term2 = len(spec_for_interp) * delta_S

        delta_f_wings = f_wings * np.sqrt((wing_term1 / tail_flux_excess)**2 +
                                          (wing_term2 / np.sum(spectrum))**2)

        sigw_term1 = (np.sum([(delta_S + g_uncert(vel)) * vel**2 for vel in
                              vels_for_interp[low_mask]]) +
                      np.sum([(delta_S + g_uncert(vel)) * vel**2 for vel in
                              vels_for_interp[high_mask]]))

        sigw_term2 = wing_term1

        delta_sigma_wing = 0.5 * sigma_wing * \
            np.sqrt((sigw_term1 / var_wing) +
                    (sigw_term2 / tail_flux_excess)**2)

        if asymm == "full":
            delta_a = asymm_val * len(spectrum) * delta_S * \
                np.sqrt((2 / (asymm_val * np.sum(spectrum)))**2 +
                        (np.sum(spectrum))**-2)
        else:

            a_term2 = wing_term1

            n_vals = len(neg_iter) + len(pos_iter)

            delta_a = asymm_val * \
                np.sqrt((2 * n_vals * delta_S / (asymm_val * tail_flux_excess))**2 +
                        (a_term2 / tail_flux_excess)**2)

        kap_term1_denom = \
            np.sum([(spec - hwhm_gauss(vel)) for spec, vel in
                    zip(spec_for_interp[fwhm_mask],
                        vels_for_interp[fwhm_mask])])

        kap_term1_numer = np.sum([delta_S + g_uncert(vel) for vel in
                                 vels_for_interp[fwhm_mask]])

        kap_term2_numer = \
            np.sum([g_uncert(vel) for vel in vels_for_interp[fwhm_mask]])

        kap_term2_denom = \
            np.sum([hwhm_gauss(vel) for vel in vels_for_interp[fwhm_mask]])

        delta_kappa = np.abs(kappa) * \
            np.sqrt((kap_term1_numer / kap_term1_denom)**2 +
                    (kap_term2_numer / kap_term2_denom)**2)

        param_stderrs = np.array([delta_sigma, delta_v_peak, delta_f_wings,
                                  delta_sigma_wing, delta_a, delta_kappa])
    else:
        param_stderrs = np.array([0.] * 6)

    params = np.array([sigma, peak_velocity, f_wings, sigma_wing, asymm_val,
                       kappa])
    param_names = ["sigma", "v_peak", "f_wings", "sigma_wing", "asymm",
                   "kappa"]

    return params, param_stderrs, param_names, hwhm_gauss


def fit_hwhm(vels, spectrum, asymm='full', sigma_noise=None, nbeams=1,
             interp_factor=10, niters=None):
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

    sigma, fwhm_points, peak_velocity = \
        find_hwhm(vels, spectrum, interp_factor)

    maxval = spectrum.max()

    # Define a Gaussian with this width
    hwhm_gauss = models.Gaussian1D(amplitude=maxval, mean=peak_velocity,
                                   stddev=sigma)

    # If niter is given, use gauss_uncert_sampler to get the parameter
    # uncertainties

    params, param_stderrs, param_names, hwhm_gauss = \
        _hwhm_fitter(vels, spectrum, hwhm_gauss, asymm=asymm,
                     sigma_noise=sigma_noise if niters is None else None,
                     nbeams=nbeams, interp_factor=interp_factor)

    if niters is not None:
        param_stderrs = \
            gauss_uncert_sampler(vels, spectrum, hwhm_gauss, params,
                                 int(niters), asymm,
                                 sigma_noise=sigma_noise * np.sqrt(nbeams),
                                 interp_factor=interp_factor)

    return params, param_stderrs, param_names, hwhm_gauss


def gauss_uncert_sampler(vels, spectrum, model, params, niters, asymm,
                         sigma_noise=None, interp_factor=10,
                         ci=[15, 85], verbose=False):
    '''
    Calculate the uncertainty in the HWHM model parameters due to the Gaussian
    profile assumptions.

    The uncertainty is estimated by:

    1) Re-sampling the spectrum within sigma_noise
    2) Changing the peak and width of the model within the expected
       uncertainty.


    '''

    chan_width = np.abs(np.diff(vels[:2])[0])

    # Error in profile width
    delta_sigma = chan_width / (2 * np.sqrt(2 * np.log(2)))

    # Error in peak velocity should approach channel width ASSUMING the
    # shuffling is optimized
    delta_v_peak = chan_width / 2.

    param_values = np.empty((niters, 6))

    spec_for_interp, vels_for_interp = reorder_spectra(vels, spectrum)

    interp1 = InterpolatedUnivariateSpline(vels_for_interp,
                                           spec_for_interp, k=3)
    interp_factor = float(interp_factor)
    chan_size = np.diff(vels_for_interp[:2])[0] / interp_factor
    upsamp_vels = np.linspace(vels_for_interp.min(),
                              vels_for_interp.max() + 0.9 * chan_size,
                              vels_for_interp.size * interp_factor)
    upsamp_spectrum = interp1(upsamp_vels)
    # Increase the noise per point in the up-sampled data
    upsamp_sigma_noise = sigma_noise * np.sqrt(interp_factor)

    for i in range(niters):

        # Re-sample spectrum values if sigma_noise is given
        if sigma_noise is not None:
            spec_resamp = upsamp_spectrum + \
                np.random.normal(0, sigma_noise,
                                 size=upsamp_spectrum.shape)
        else:
            spec_resamp = upsamp_spectrum

        new_peak = upsamp_spectrum.max()
        new_mean = model.mean + np.random.normal(0, delta_v_peak)
        new_sigma = model.stddev + np.random.normal(0, delta_sigma)
        # new_sigma, fwhm_points_, new_mean = \
        #     find_hwhm(vels, spectrum, interp_factor)

        pert_model = models.Gaussian1D(amplitude=new_peak, mean=new_mean,
                                       stddev=new_sigma)

        pert_params = _hwhm_fitter(upsamp_vels, spec_resamp, pert_model,
                                   asymm=asymm,
                                   sigma_noise=None,
                                   interp_factor=interp_factor)[0]
        param_values[i] = pert_params

    lower_lim = params - np.nanpercentile(param_values, ci[0], axis=0)
    upper_lim = np.nanpercentile(param_values, ci[1], axis=0) - params

    # Insert the assumed known errors in sigma and vpeak
    lower_lim[0] = upper_lim[0] = delta_sigma
    lower_lim[1] = upper_lim[1] = delta_v_peak

    if verbose:
        from astropy.visualization import hist
        import matplotlib.pyplot as p
        p.subplot(321)
        _ = hist(param_values[:, 0])
        p.axvline(params[0], color='g')
        p.axvline(params[0] - lower_lim[0], color='r')
        p.axvline(params[0] + upper_lim[0], color='r')
        p.subplot(322)
        _ = hist(param_values[:, 1])
        p.axvline(params[1], color='g')
        p.axvline(params[1] - lower_lim[1], color='r')
        p.axvline(params[1] + upper_lim[1], color='r')
        p.subplot(323)
        _ = hist(param_values[:, 2])
        p.axvline(params[2], color='g')
        p.axvline(params[2] - lower_lim[2], color='r')
        p.axvline(params[2] + upper_lim[2], color='r')
        p.subplot(324)
        _ = hist(param_values[:, 3])
        p.axvline(params[3], color='g')
        p.axvline(params[3] - lower_lim[3], color='r')
        p.axvline(params[3] + upper_lim[3], color='r')
        p.subplot(325)
        _ = hist(param_values[:, 4])
        p.axvline(params[4], color='g')
        p.axvline(params[4] - lower_lim[4], color='r')
        p.axvline(params[4] + upper_lim[4], color='r')
        p.subplot(326)
        _ = hist(param_values[:, 5])
        p.axvline(params[5], color='g')
        p.axvline(params[5] - lower_lim[5], color='r')
        p.axvline(params[5] + upper_lim[5], color='r')
        p.draw()
        raw_input("?")
        p.clf()

    return np.vstack([lower_lim, upper_lim])


def gauss_uncert(vel, model, chan_width, nbeams, sigma_noise):
    '''
    Return the uncertainty of the assumed HWHM Gaussian. Assumes errors to be
    propagated in quadrature and without correlations.
    '''

    mod_val = model(vel)

    # Uncertainties in v_peak and sigma assumed to arise solely from the
    # finite velocity resolution -- taken at half the channel width

    sigma = model.stddev.value
    v_peak = model.mean.value
    amp = model.amplitude.value

    diff_vel = (vel - v_peak)

    # Width uncertainty
    term1 = (diff_vel**2 * chan_width / sigma**3)**2 / (8 * np.log(2))

    # mean uncertainty
    term2 = (0.5 * diff_vel * chan_width / sigma**2)**2

    # amp uncertainty
    term3 = nbeams * (sigma_noise / amp)**2

    return mod_val * np.sqrt(term1 + term2 + term3)
