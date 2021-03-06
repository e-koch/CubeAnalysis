
'''
Basic models for decomposing large-scale stacked profiles
'''

import numpy as np
from astropy.modeling import models, fitting
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import erf
from scipy.optimize import curve_fit
from functools import partial
from astropy.convolution import convolve

from .spectral_fitting import gauss_model_discrete


def fit_2gaussian(vels, spectrum):
    '''
    Fit a 2 Gaussian model with the means tied.

    .. todo:: Allow passing sigma for weighting in the fit.

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


def fit_gaussian(vels, spectrum, p0=None, sigma=None, use_discrete=False,
                 kernel=None, add_chan_width_err=True):
    '''
    Fit a Gaussian model.

    Parameters
    ----------
    sigma : `~astropy.units.Quantity`, optional
        Pass a single value to be used for all points or an array of values
        equal to the size of vels and spectrum. These are used to define
        weights in the fit, and the weights as defined as 1 / sigma, as is
        used in `~scipy.optimize.curve_fit`.

    '''

    if hasattr(vels, 'unit'):
        vel_unit = vels.unit
    else:
        vel_unit = 1.

    if hasattr(vels, 'unit'):
        spec_unit = spectrum.unit
    else:
        spec_unit = 1.

    if p0 is None:
        max_vel = vels[np.argmax(spectrum)]
        specmax = spectrum.max()

        # Estimate the inner width from the HWHM
        sigma_est = find_hwhm(vels, spectrum)[0]

        p0 = (specmax.value, max_vel.value, sigma_est)

    else:
        specmax, max_vel, sigma_est = p0

    if sigma is not None:
        if hasattr(sigma, 'size'):
            if sigma.size > 1:
                if sigma.size != vels.size:
                    raise ValueError("sigma must match the data shape when "
                                     "multiple values are given.")
                weights = 1 / np.abs(sigma.value)
            else:
                # A quantity will still have a `size` attribute with one
                # element
                weights = 1 / np.array([np.abs(sigma.value)] * len(vels))
        else:
            weights = 1 / np.array([np.abs(sigma)] * len(vels))
    else:
        weights = None

    # Use this as the narrow estimate, and let the wide guess be 3x

    g_init = models.Gaussian1D(amplitude=specmax, mean=max_vel,
                               stddev=sigma_est)

    if use_discrete:

        # Don't use the astropy fitting for the discrete model
        # Though the astropy fitter is using something quite similar
        # to this, anyways.

        if kernel is not None:

            def spectral_model(vels, amp, mean, stddev):

                model = gauss_model_discrete(vels, amp=amp, stddev=stddev,
                                             mean=mean)
                return convolve(model, kernel)

        else:
            def spectral_model(vels, amp, mean, stddev):

                return gauss_model_discrete(vels, amp=amp, stddev=stddev,
                                            mean=mean)

        out = curve_fit(spectral_model,
                        vels, spectrum, p0=p0,
                        sigma=sigma * np.ones_like(vels),
                        absolute_sigma=True, maxfev=100000)

        g_fit = models.Gaussian1D(amplitude=out[0][0] * spec_unit,
                                  mean=out[0][1] * vel_unit,
                                  stddev=out[0][2] * vel_unit)

        cov = out[1]
        parerrs = np.sqrt(np.diag(out[1]))

    else:

        fit_g = fitting.LevMarLSQFitter()

        g_fit = fit_g(g_init, vels, spectrum, weights=weights)

        # The covariance matrix is hidden away... tricksy
        cov = fit_g.fit_info['param_cov']
        if cov is None:
            cov = np.zeros((3, 3)) * np.NaN

    parnames = g_fit.param_names
    parvals = g_fit.parameters

    if cov is not None:
        chan_width = np.diff(vels[:2])[0].value
        parerrs = []
        for name, var, val in zip(parnames, np.diag(cov), parvals):
            # print(name, var, val)
            if "mean" in name or "stddev" in name and add_chan_width_err:
                # Add the finite channel width in quadrature
                # print(var, chan_width)
                # print(np.diag(cov))
                stderr = np.sqrt(var + (0.35 * chan_width)**2)
            else:
                stderr = np.sqrt(var)

            parerrs.append(stderr)
    else:
        parerrs = [np.NaN] * len(parnames)

    # Sometimes the width is negative
    parvals[-1] = np.abs(parvals[-1])

    return parvals, parerrs, cov, parnames, g_fit


def reorder_spectra(vels, spectrum):
    spec_for_interp = spectrum if np.diff(vels[:2])[0] > 0 else spectrum[::-1]
    vels_for_interp = vels if np.diff(vels[:2])[0] > 0 else vels[::-1]

    return spec_for_interp, vels_for_interp


def find_hwhm(vels, spectrum, interp_factor=10):
    '''
    Return the equivalent Gaussian sigma based on the HWHM positions.
    '''

    fwhm_points, upsamp_vels, interp1 = \
        find_peak_window(vels, spectrum, interp_factor=interp_factor,
                         peak_fraction=0.5, return_upsampled=True)

    fwhm = fwhm_points[1] - fwhm_points[0]

    # Convert to equivalent Gaussian sigma
    sigma = fwhm / np.sqrt(8 * np.log(2))

    upsamp_spec = interp1(upsamp_vels)
    peak_velocity = upsamp_vels[np.argmax(upsamp_spec)]

    return sigma, fwhm_points, peak_velocity


def find_peak_window(vels, spectrum, interp_factor=10, peak_fraction=0.5,
                     return_upsampled=False):
    '''
    Define a spectral window around the peak defined by the
    peak fraction given.
    '''

    # Model the spectrum with a spline
    # x values must be increasing for the spline, so flip if needed.
    spec_for_interp, vels_for_interp = reorder_spectra(vels, spectrum)

    interp1 = InterpolatedUnivariateSpline(vels_for_interp,
                                           spec_for_interp, k=3)

    # Upsample in velocity to estimate the peak position
    interp_factor = float(interp_factor)
    chan_size = np.diff(vels_for_interp[:2])[0] / interp_factor
    upsamp_vels = np.linspace(vels_for_interp.min(),
                              vels_for_interp.max() + 0.9 * chan_size,
                              vels_for_interp.size * interp_factor)

    halfmax = interp1(upsamp_vels).max() * peak_fraction

    interp2 = InterpolatedUnivariateSpline(vels_for_interp,
                                           spec_for_interp - halfmax, k=3)

    fwhm_points = interp2.roots()
    if len(fwhm_points) < 2:
        raise ValueError("Found less than 2 roots!")
    # Only keep the min/max if there are multiple
    fwhm_points = (min(fwhm_points), max(fwhm_points))

    if return_upsampled:
        return fwhm_points, upsamp_vels, interp1

    return fwhm_points


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

    tail_flux_excess_low = \
        np.sum([spec - hwhm_gauss(vel) for spec, vel in
                zip(spec_for_interp[low_mask], vels_for_interp[low_mask])])
    tail_flux_excess_high = \
        np.sum([spec - hwhm_gauss(vel) for spec, vel in
                zip(spec_for_interp[high_mask], vels_for_interp[high_mask])])

    tail_flux_excess = tail_flux_excess_low + tail_flux_excess_high

    # Calculate fraction in the wings
    f_wings = tail_flux_excess / np.sum(spectrum)

    # Calculate the symmetric and asymmetric line wing fractions
    # Symmetric is defined as the same ratio as the full f_wing, but only
    # computed for whichever side has less flux.
    # The asymmetric component is the difference b/w the full f_wing and
    # the symmetric f_wing
    # if tail_flux_excess_low > tail_flux_excess_high:
    #     f_wings_symm = tail_flux_excess_high / \
    #         np.sum(spec_for_interp[vels_for_interp > fwhm_points[1]])
    # else:
    #     f_wings_symm = tail_flux_excess_low / \
    #         np.sum(spec_for_interp[vels_for_interp < fwhm_points[0]])

    # f_wings_asymm = f_wings - f_wings_symm

    # Equivalent sigma^2 of the wings
    var_wing = (np.sum([vel**2 * (spec - hwhm_gauss(vel)) for spec, vel in
                        zip(spec_for_interp[low_mask],
                            vels_for_interp[low_mask])]) +
                np.sum([vel**2 * (spec - hwhm_gauss(vel)) for spec, vel in
                        zip(spec_for_interp[high_mask],
                            vels_for_interp[high_mask])]))

    sigma_wing = np.sqrt(np.abs(var_wing / tail_flux_excess))

    if asymm == "full":
        # neg_iter = range(maxpos - 1, -1, -1)
        # pos_iter = range(maxpos + 1, len(vels))
        # asymm_val = np.sum([np.sqrt((spec_for_interp[vel_l] -
        #                              spec_for_interp[vel_u])**2)
        #                     for vel_l, vel_u in zip(neg_iter, pos_iter)]) / \
        #     np.sum(spectrum)
        left_sum = np.sum(spec_for_interp[vels_for_interp < peak_velocity])
        right_sum = np.sum(spec_for_interp[vels_for_interp > peak_velocity])
        asymm_val = (right_sum - left_sum) / np.sum(spec_for_interp)
    elif asymm == 'wings':
        # neg_iter = np.where(vels_for_interp < fwhm_points[0])[0]
        # pos_iter = np.where(vels_for_interp > fwhm_points[1])[0]
        # asymm_val = np.sum([np.sqrt((spec_for_interp[vel_l] -
        #                              spec_for_interp[vel_u])**2)
        #                     for vel_l, vel_u in zip(neg_iter, pos_iter)]) / \
        #     tail_flux_excess
        left_sum = np.sum(spec_for_interp[vels_for_interp < fwhm_points[0]] -
                          hwhm_gauss(vels_for_interp[vels_for_interp < fwhm_points[0]]))
        right_sum = np.sum(spec_for_interp[vels_for_interp > fwhm_points[1]] -
                           hwhm_gauss(vels_for_interp[vels_for_interp > fwhm_points[1]]))
        asymm_val = (right_sum - left_sum) / tail_flux_excess

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

        chan_width = np.abs(np.diff(vels[:2])[0])

        # Set to 1-sigma area of a rectangular channel.
        delta_v_peak = 0.35 * chan_width

        # Error in profile width
        # Similar assumptions as for the peak location
        delta_sigma = 0.35 * chan_width

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
            # delta_a = asymm_val * len(spectrum) * delta_S * \
            #     np.sqrt((2 / (asymm_val * np.sum(spectrum)))**2 +
            #             (np.sum(spectrum))**-2)
            delta_a = np.abs(asymm_val) * delta_S * \
                np.sqrt(((vels_for_interp < peak_velocity).sum() / left_sum)**2 +
                        ((vels_for_interp > peak_velocity).sum() / right_sum)**2)
        else:

            # a_term2 = wing_term1
            # n_vals = len(neg_iter) + len(pos_iter)
            # delta_a = asymm_val * \
            #     np.sqrt((2 * n_vals * delta_S / (asymm_val * tail_flux_excess))**2 +
            #             (a_term2 / tail_flux_excess)**2)

            lt_peak = vels_for_interp < peak_velocity
            gt_peak = vels_for_interp > peak_velocity

            delta_f_v_lt_vpeak = \
                np.sum([delta_S + g_uncert(vel) for vel in
                        vels_for_interp[lt_peak]]) / tail_flux_excess_low
            delta_f_v_gt_vpeak = \
                np.sum([delta_S + g_uncert(vel) for vel in
                        vels_for_interp[gt_peak]]) / tail_flux_excess_high

            delta_a = np.abs(delta_f_v_lt_vpeak) + np.abs(delta_f_v_gt_vpeak)

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
             interp_factor=10, niters=None, verbose=False):
    '''
    Scale the inner Gaussian to the HWHM of the profile.

    Extracts the equivalent inner gaussian width, the fraction of flux in the
    wings, the equivalent width of the wings, and the asymmetry of the wings or
    profile. Each of these is defined in Stilp et al. (2013).
    https://ui.adsabs.harvard.edu/#abs/2013ApJ...765..136S/abstract

    The asymmetry parameter is redefined to be the difference of the flux
    above and below the peak velocity divided by the total flux.

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
        if sigma_noise is not None:
            samp_noise = sigma_noise * np.sqrt(nbeams)
        else:
            samp_noise = None

        param_stderrs, samps = \
            gauss_uncert_sampler(vels, spectrum, hwhm_gauss, params,
                                 int(niters), asymm,
                                 sigma_noise=samp_noise,
                                 interp_factor=interp_factor,
                                 verbose=verbose, return_samps=True)

        return params, param_stderrs, param_names, hwhm_gauss, samps

    return params, param_stderrs, param_names, hwhm_gauss


def gauss_uncert_sampler(vels, spectrum, model, params, niters, asymm,
                         sigma_noise=None, interp_factor=10,
                         ci=[15, 85], verbose=False, return_samps=False):
    '''
    Calculate the uncertainty in the HWHM model parameters due to the Gaussian
    profile assumptions.

    The uncertainty is estimated by:

    1) Re-sampling the spectrum within sigma_noise
    2) Changing the peak and width of the model within the expected
       uncertainty.


    '''

    chan_width = np.abs(np.diff(vels[:2])[0])

    # Error in peak velocity should approach channel width ASSUMING the
    # shuffling is optimized.
    # To make this an approximate Gaussian error, take sigma ~ 0.35 * channel
    # This assumes that we can confidently determine the peak to within 1
    # channel, so +/-0.35 * channel is an 'equivalent' Gaussian interval.
    # This is made more reasonable in that the location of the peak is
    # also determined by the relative heights of the surrounding channels,
    # but since an explicity fit is not done, we don't have a great way
    # of including that information here.
    delta_v_peak = 0.35 * chan_width

    # Error in profile width
    # Similar assumptions as for the peak location
    delta_sigma = 0.35 * chan_width

    param_values = np.empty((niters, 6))

    for i in range(niters):

        # Re-sample spectrum values if sigma_noise is given
        if sigma_noise is not None:
            spec_resamp = spectrum + \
                np.random.normal(0, sigma_noise,
                                 size=spectrum.shape)
        else:
            spec_resamp = spectrum

        new_peak = spec_resamp.max()
        new_sigma, fwhm_points_, new_mean = \
            find_hwhm(vels, spec_resamp, interp_factor)

        # Due to the finite bins, there's a limit to how well we know the
        # mean and sigma. Reflect this uncertainty by sampling for
        # value of the mean and sigma
        new_mean = new_mean + np.random.normal(0, delta_v_peak)
        new_sigma = new_sigma + np.random.normal(0, delta_sigma)

        pert_model = models.Gaussian1D(amplitude=new_peak, mean=new_mean,
                                       stddev=new_sigma)

        pert_params = _hwhm_fitter(vels, spec_resamp, pert_model,
                                   asymm=asymm,
                                   sigma_noise=None,
                                   interp_factor=interp_factor)[0]
        param_values[i] = pert_params

    lower_lim = params - np.nanpercentile(param_values, ci[0], axis=0)
    upper_lim = np.nanpercentile(param_values, ci[1], axis=0) - params

    # Insert the assumed known errors in sigma and vpeak
    # lower_lim[0] = upper_lim[0] = delta_sigma
    # lower_lim[1] = upper_lim[1] = delta_v_peak

    if verbose:
        from astropy.visualization import hist
        import matplotlib.pyplot as p
        p.subplot(321)
        _ = hist(param_values[:, 0], bins='scott')
        p.axvline(params[0], color='g')
        p.axvline(params[0] - lower_lim[0], color='r')
        p.axvline(params[0] + upper_lim[0], color='r')
        p.xlabel("sigma")
        p.subplot(322)
        _ = hist(param_values[:, 1], bins='scott')
        p.axvline(params[1], color='g')
        # p.axvline(params[1] - lower_lim[1], color='r')
        # p.axvline(params[1] + upper_lim[1], color='r')
        p.xlabel("v_peak")
        p.subplot(323)
        _ = hist(param_values[:, 2], bins='scott')
        p.axvline(params[2], color='g')
        p.axvline(params[2] - lower_lim[2], color='r')
        p.axvline(params[2] + upper_lim[2], color='r')
        p.xlabel("f_wings")
        p.subplot(324)
        _ = hist(param_values[:, 3], bins='scott')
        p.axvline(params[3], color='g')
        p.axvline(params[3] - lower_lim[3], color='r')
        p.axvline(params[3] + upper_lim[3], color='r')
        p.xlabel("sigma_wing")
        p.subplot(325)
        _ = hist(param_values[:, 4], bins='scott')
        p.axvline(params[4], color='g')
        p.axvline(params[4] - lower_lim[4], color='r')
        p.axvline(params[4] + upper_lim[4], color='r')
        p.xlabel("asymm")
        p.subplot(326)
        _ = hist(param_values[:, 5], bins='scott')
        p.axvline(params[5], color='g')
        p.axvline(params[5] - lower_lim[5], color='r')
        p.axvline(params[5] + upper_lim[5], color='r')
        p.xlabel("kappa")
        p.draw()
        raw_input("?")
        p.clf()

    if return_samps:
        return np.vstack([lower_lim, upper_lim]), param_values

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


def find_linewing_asymm(vels, spec_n, spec_s, interp_factor=2,
                        niters=None, ci=[15, 85],
                        sigma_noise_s=None,
                        sigma_noise_n=None):
    '''
    Find the symmetric and asymmetric line wing fraction from stacked profiles
    split into two halves of some region (i.e., N/S halves of a galaxy,
    different sides of an outflow, etc.). Defined with respect to a Gaussian
    model for the central peak of the stacked profiles
    '''

    spec_n_r = reorder_spectra(vels.copy(), spec_n)[0]
    spec_s_r, vels = reorder_spectra(vels.copy(), spec_s)

    tot_spec = spec_n_r + spec_s_r

    # Fit w/o finding errors
    parvals_hwhm, parerrs_hwhm, parnames_hwhm, hwhm_gauss = \
        fit_hwhm(vels, tot_spec,
                 niters=None, interp_factor=2.)

    hwhm_factor = np.sqrt(2 * np.log(2))
    fwhm_points = [hwhm_gauss.mean - hwhm_gauss.stddev * hwhm_factor,
                   hwhm_gauss.mean + hwhm_gauss.stddev * hwhm_factor]

    low_mask = vels < fwhm_points[0]
    high_mask = vels > fwhm_points[1]

    blue_excess = np.sum((spec_s_r - spec_n_r)[low_mask])
    red_excess = np.sum((spec_n_r - spec_s_r)[high_mask])

    tail_flux_excess_low = \
        np.sum([spec - hwhm_gauss(vel) for spec, vel in
                zip(tot_spec[low_mask], vels[low_mask])])
    tail_flux_excess_high = \
        np.sum([spec - hwhm_gauss(vel) for spec, vel in
                zip(tot_spec[high_mask], vels[high_mask])])

    tail_flux_excess = tail_flux_excess_low + tail_flux_excess_high

    # Calculate fraction in the wings
    f_wings = tail_flux_excess / np.sum(tot_spec)

    f_symm = (tail_flux_excess - blue_excess - red_excess) / np.sum(tot_spec)

    f_asymm = (blue_excess + red_excess) / np.sum(tot_spec)

    params = np.array([f_wings, f_symm, f_asymm])

    # Now calculate uncertainty
    chan_width = np.abs(np.diff(vels[:2])[0])

    # Error in profile width
    delta_sigma = chan_width / (2 * np.sqrt(2 * np.log(2)))

    # Error in peak velocity should approach channel width ASSUMING the
    # shuffling is optimized
    delta_v_peak = chan_width / 2.

    if niters is not None:

        param_values = np.empty((niters, 3))

        for i in range(niters):

            # Re-sample spectrum values if sigma_noise is given
            if sigma_noise_n is not None:
                spec_n_resamp = spec_n_r + \
                    np.random.normal(0, sigma_noise_n,
                                     size=spec_n_r.shape)
            else:
                spec_n_resamp = spec_n_r

            if sigma_noise_s is not None:
                spec_s_resamp = spec_s_r + \
                    np.random.normal(0, sigma_noise_s,
                                     size=spec_s_r.shape)
            else:
                spec_s_resamp = spec_s_r

            tot_spec_resamp = spec_n_resamp + spec_s_resamp

            new_peak = tot_spec_resamp.max()
            new_mean = hwhm_gauss.mean + np.random.normal(0, delta_v_peak)
            new_sigma = hwhm_gauss.stddev + np.random.normal(0, delta_sigma)

            pert_model = models.Gaussian1D(amplitude=new_peak, mean=new_mean,
                                           stddev=new_sigma)

            fwhm_points = [pert_model.mean - pert_model.stddev * hwhm_factor,
                           pert_model.mean + pert_model.stddev * hwhm_factor]

            low_mask = vels < fwhm_points[0]
            high_mask = vels > fwhm_points[1]

            blue_excess_pert = np.sum((spec_s_resamp - spec_n_resamp)[low_mask])
            red_excess_pert = np.sum((spec_n_resamp - spec_s_resamp)[high_mask])

            tail_flux_excess_low_pert = \
                np.sum([spec - pert_model(vel) for spec, vel in
                        zip(tot_spec_resamp[low_mask], vels[low_mask])])
            tail_flux_excess_high_pert = \
                np.sum([spec - pert_model(vel) for spec, vel in
                        zip(tot_spec_resamp[high_mask], vels[high_mask])])

            tail_flux_excess_pert = tail_flux_excess_low_pert + \
                tail_flux_excess_high_pert

            # Calculate fraction in the wings
            f_wings_pert = tail_flux_excess_pert / np.sum(tot_spec_resamp)

            f_symm_pert = (tail_flux_excess_pert - blue_excess_pert -
                           red_excess_pert) / np.sum(tot_spec_resamp)

            f_asymm_pert = (blue_excess_pert + red_excess_pert) / \
                np.sum(tot_spec_resamp)

            param_values[i] = np.array([f_wings_pert, f_symm_pert, f_asymm_pert])

        lower_lim = params - np.nanpercentile(param_values, ci[0], axis=0)
        upper_lim = np.nanpercentile(param_values, ci[1], axis=0) - params

        return params, lower_lim, upper_lim

    else:
        return params
