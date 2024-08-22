
import numpy as np
import numpy.testing as npt
from astropy.modeling import models

from ..spectral_stacking_models import (fit_gaussian, fit_2gaussian, fit_hwhm,
                                        find_linewing_asymm)


np.random.seed(347895209)

def gauss_with_noise(vels, amp, mean, stddev, noise_level,
                     return_models=False):

    model = models.Gaussian1D(amp, mean, stddev)

    if noise_level == 0.0:
        out_vals = model(vels)

        if return_models:
            return out_vals, model
        else:
            return out_vals
    else:
        out_vals = model(vels) + \
            np.random.normal(0, noise_level, size=vels.shape)
        if return_models:
            return out_vals, model
        else:
            return out_vals


def gauss_with_wings_noise(vels, amp, mean, stddev, noise_level,
                           asymm=0.0,
                           return_models=False):
    '''
    Note that the asymm parameter isn't generalized. It only decimates the
    line wings on one-side, depending on whether it is positive or negative.
    '''

    mod = models.Gaussian1D(amp, mean, stddev)(vels)
    lor = models.Lorentz1D(amp, mean, stddev * 2.35)(vels)

    # Only use the Lorentzian wings
    hwhm = np.sqrt(2 * np.log(2))
    lor[(vels > mean - stddev * hwhm) & (vels < mean + stddev * hwhm)] = 0.0
    mod[vels < mean - stddev * hwhm] = 0.0
    mod[vels > mean + stddev * hwhm] = 0.0

    # Allow the line wings to be asymmetric
    if asymm == 0.0:
        clean_vals = mod + lor
    else:
        # Apply a exponential weighting to one side
        # Set the slope of the exponential based on the given asymmetry
        # The integral from HWHM to inf (+/-) equals |asymm|
        alpha = 1 + 0.5 * amp * (mean + hwhm * stddev) / np.abs(asymm)

        if asymm < 0.:
            # blue_weight = models.PowerLaw1D(0.5 * amp, mean + hwhm * stddev,
            #                                 alpha)
            # red_weight = lambda x: np.array([1.0])
             blue_weight = -asymm
             red_weight = 1.0

        else:
            # blue_weight = lambda x: np.array([1.0])
            # red_weight = models.PowerLaw1D(0.5 * amp, mean + hwhm * stddev,
            #                                alpha)
            blue_weight = 1.0
            red_weight = asymm

        # Reverse the exponential curve for the red side
        # red_wing = lor * (vels < mean - stddev * hwhm) * red_weight(np.abs(vels))[::-1]
        # blue_wing = lor * (vels > mean + stddev * hwhm) * blue_weight(np.abs(vels))[::-1]
        red_wing = lor * (vels < mean - stddev * hwhm) * red_weight
        blue_wing = lor * (vels > mean + stddev * hwhm) * blue_weight

        clean_vals = mod + red_wing + blue_wing

    if noise_level == 0.0:
        # Return an unaltered Gaussian component
        mod = models.Gaussian1D(amp, mean, stddev)(vels)

        if return_models:
            return clean_vals, mod, lor

        return clean_vals

    noisy_mod = clean_vals + \
        np.random.normal(0, noise_level, size=vels.shape)

    if return_models:
        # Return an unaltered Gaussian component
        mod = models.Gaussian1D(amp, mean, stddev)(vels)
        return noisy_mod, mod, lor

    return noisy_mod


def twogauss_with_noise(vels, amp, mean, stddev, amp1, mean1, stddev1,
                        noise_level):

    model_vals = models.Gaussian1D(amp, mean, stddev)(vels) + \
        models.Gaussian1D(amp1, mean1, stddev1)(vels)

    if noise_level == 0.0:
        return model_vals

    return model_vals + \
        np.random.normal(0, noise_level, size=vels.shape)


def test_twogauss_model():

    vels = np.linspace(-1, 1, 100)

    twogauss_spectrum = \
        twogauss_with_noise(vels, 0.5, 0.0, 0.05, 0.5, 0.0, 0.3, 0.01)

    parvals, parerrs, cov, parnames, g_HI = \
        fit_2gaussian(vels, twogauss_spectrum)

    actual_params = [0.5, 0.0, 0.05, 0.5, 0.3]

    for fit, act, err in zip(parvals, actual_params, parerrs):
        npt.assert_allclose(act, fit, atol=2 * err)


def assert_between(act, val, low_err, up_err):

    assert (act >= val - low_err) & (act <= val + up_err)


def test_hwhm_model():

    vels = np.linspace(-1, 1, 100)

    model, gauss, lor = \
        gauss_with_wings_noise(vels, 1., 0.0, 0.1, 0.0, return_models=True)

    noisy_model = \
        gauss_with_wings_noise(vels, 1., 0.0, 0.1, 0.01)

    parvals, parerrs, param_names, hwhm_gauss = \
        fit_hwhm(vels, noisy_model, sigma_noise=0.01,
                 niters=100)[:-1]

    parvals_clean, parerrs_clean = \
        fit_hwhm(vels, model, sigma_noise=0.01,
                 niters=100)[:2]

    f_wings = np.sum((model - gauss)) / np.sum(model)
    sigma_wings = np.sqrt(np.sum([(m - g) * v**2 for m, g, v in
                                  zip(model, gauss, vels)]) /
                          np.sum((model - gauss)))

    asymm = 0.0
    kappa = 0.0

    actual_params = [0.1, 0.0, f_wings, sigma_wings, asymm, kappa]

    # Compare the clean parameter estimates
    for act, val, low, up in zip(actual_params, parvals_clean, *parerrs_clean):
        assert_between(act, val, low, up)

    # Compare the noisy parameter estimates
    for act, val, low, up in zip(actual_params, parvals, *parerrs):
        assert_between(act, val, low, up)


def test_hwhm_model_wgauss():

    vels = np.linspace(-1, 1, 100)

    model, gauss = \
        gauss_with_noise(vels, 1., 0.0, 0.1, 0.0, return_models=True)

    noisy_model = \
        gauss_with_noise(vels, 1., 0.0, 0.1, 0.01)

    parvals, parerrs, param_names, hwhm_gauss = \
        fit_hwhm(vels, noisy_model, sigma_noise=0.01,
                 niters=100)[:-1]

    parvals_clean, parerrs_clean = \
        fit_hwhm(vels, model, sigma_noise=0.01,
                 niters=100)[:2]

    f_wings = 0.0
    sigma_wings = 0.0
    asymm = 0.0
    kappa = 0.0

    actual_params = [0.1, 0.0, f_wings, sigma_wings, asymm, kappa]
    names = ['sigma', 'vcen', 'fwing', 'sig_wing', 'asymm', 'kappa']

    # Compare the clean parameter estimates
    for name, act, val, low, up in zip(names, actual_params, parvals_clean,
                                       *parerrs_clean):
        if name == 'sig_wing':
            continue
        assert_between(act, val, low, up)

    # Compare the noisy parameter estimates
    for name, act, val, low, up in zip(names, actual_params, parvals,
                                       *parerrs):
        if name == 'sig_wing':
            continue
        assert_between(act, val, low, up)


def test_hwhm_model_asymmwings():

    vels = np.linspace(-1, 1, 100)

    model, gauss, lor = \
        gauss_with_wings_noise(vels, 1., 0.0, 0.1, 0.0, return_models=True,
                               asymm=-0.85)

    noisy_model = \
        gauss_with_wings_noise(vels, 1., 0.0, 0.1, 0.01,
                               asymm=-0.85)

    parvals, parerrs, param_names, hwhm_gauss = \
        fit_hwhm(vels, noisy_model, sigma_noise=0.01,
                 niters=100)[:-1]

    parvals_clean, parerrs_clean = \
        fit_hwhm(vels, model, sigma_noise=0.01,
                 niters=100)[:2]

    high_mask = vels > 0.0 + 0.1 * np.sqrt(2 * np.log(2))
    low_mask = vels < 0.0 - 0.1 * np.sqrt(2 * np.log(2))

    # peak_high_mask = np.logical_and(vels > 0.,
    #                                 vels < 0.0 + 0.1 * np.sqrt(2 * np.log(2)))
    # peak_low_mask = np.logical_and(vels < 0.,
    #                                vels > 0.0 - 0.1 * np.sqrt(2 * np.log(2)))

    f_wings = (np.sum(model[low_mask] - gauss[low_mask]) +
               np.sum(model[high_mask] - gauss[high_mask])) / np.sum(model)
    sigma_wings = np.NaN

    pos_mask = vels > 0
    neg_mask = vels < 0

    asymm = np.sum(model[pos_mask] - model[neg_mask]) / np.sum(model)
    kappa = 0.0

    actual_params = [0.1, 0.0, f_wings, sigma_wings, asymm, kappa]
    names = ['sigma', 'vcen', 'fwing', 'sig_wing', 'asymm', 'kappa']

    # Not comparing the sigma_wings b/c we aren't using it in any analysis
    # and the definition doesn't seem well-suited to substantial asymmetry

    # Compare the clean parameter estimates
    for name, act, val, low, up in zip(names, actual_params, parvals_clean,
                                       *parerrs_clean):
        if name == 'sig_wing':
            continue
        assert_between(act, val, low, up)

    # Compare the noisy parameter estimates
    for name, act, val, low, up in zip(names, actual_params, parvals,
                                       *parerrs):
        if name == 'sig_wing':
            continue
        assert_between(act, val, low, up)


def test_symm_asymm_fwing():
    '''
    Use above asymmetric setup.
    '''
    vels = np.linspace(-1, 1, 100)

    model, gauss, lor = \
        gauss_with_wings_noise(vels, 1., 0.0, 0.1, 0.0, return_models=True,
                               asymm=-0.85)

    noisy_model = \
        gauss_with_wings_noise(vels, 1., 0.0, 0.1, 0.01,
                               asymm=-0.85)
    s_model = model
    n_model = model[::-1]

    tot_model = n_model + s_model

    params_clean = find_linewing_asymm(vels, n_model, s_model)

    s_model_noisy = noisy_model
    n_model_noisy = noisy_model[::-1]

    params, low_lim, up_lim = \
        find_linewing_asymm(vels, n_model_noisy, s_model_noisy,
                            niters=100,
                            sigma_noise_n=0.01,
                            sigma_noise_s=0.01)

    high_mask = vels > 0.0 + 0.1 * np.sqrt(2 * np.log(2))
    low_mask = vels < 0.0 - 0.1 * np.sqrt(2 * np.log(2))

    # We added the model twice, so multiple gauss by 2
    f_wings = (np.sum(tot_model[low_mask] - 2 * gauss[low_mask]) +
               np.sum(tot_model[high_mask] - 2 * gauss[high_mask])) / \
        np.sum(tot_model)

    blue_excess = np.sum(s_model[low_mask] - n_model[low_mask])
    red_excess = np.sum(n_model[high_mask] - s_model[high_mask])

    f_asymm = (blue_excess + red_excess) / np.sum(tot_model)

    f_symm = f_wings - f_asymm

    npt.assert_allclose(f_wings, params_clean[0], atol=0.005)

    assert_between(f_wings, params[0], low_lim[0], up_lim[0])

    npt.assert_allclose(f_symm, params_clean[1], atol=0.005)

    assert_between(f_symm, params[1], low_lim[1], up_lim[1])

    npt.assert_allclose(f_asymm, params_clean[2], atol=0.005)

    # Errors seem to be a bit more asymmetric for f_asymm. Allow
    # a larger range
    avg_sig = 0.5 * ((params[2] - low_lim[2]) + (up_lim[2] - params[2]))
    print(f_wings, f_symm, f_asymm)
    print(params, low_lim, up_lim)
    assert_between(f_asymm, params[2], low_lim[2] - 0.5 * avg_sig,
                   up_lim[2] + 0.5 * avg_sig)
