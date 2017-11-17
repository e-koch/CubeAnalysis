
import numpy as np
import numpy.testing as npt
from astropy.modeling import models

from ..spectral_stacking_models import (fit_gaussian, fit_2gaussian, fit_hwhm)


def gauss_with_noise(vels, amp, mean, stddev, noise_level):

    if noise_level == 0.0:
        return models.Gaussian1D(amp, mean, stddev)(vels)

    return models.Gaussian1D(amp, mean, stddev)(vels) + \
        np.random.normal(0, noise_level, size=vels.shape)


def gauss_with_wings_noise(vels, amp, mean, stddev, noise_level,
                           return_models=False):

    mod = models.Gaussian1D(amp, mean, stddev)(vels)
    lor = models.Lorentz1D(amp, mean, stddev * 2.35)(vels)

    # Only use the Lorentzian wings
    hwhm = np.sqrt(2 * np.log(2))
    lor[(vels > mean - stddev * hwhm) & (vels < mean + stddev * hwhm)] = 0.0
    mod[vels < mean - stddev * hwhm] = 0.0
    mod[vels > mean + stddev * hwhm] = 0.0

    clean_vals = mod + lor

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

    assert (act > val - low_err) & (act < val + up_err)


def test_hwhm_model():

    vels = np.linspace(-1, 1, 100)

    model, gauss, lor = \
        gauss_with_wings_noise(vels, 1., 0.0, 0.1, 0.0, return_models=True)

    noisy_model = \
        gauss_with_wings_noise(vels, 1., 0.0, 0.1, 0.01)

    parvals, parerrs, param_names, hwhm_gauss = \
        fit_hwhm(vels, noisy_model, sigma_noise=0.01,
                 niters=100)

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
