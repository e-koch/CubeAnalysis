
import pytest
import numpy as np
import numpy.testing as npt
from astropy.modeling import models

from ..spectral_fitting import (sample_at_channels, gauss_model_discrete)
from ..spectral_fitting.discrete_sampler import gauss_weighted_avg


@pytest.mark.parametrize('chan_width_stddev', [0.1, 0.25, 0.5, 0.75])
def test_discrete_sampler(chan_width_stddev):

    amp = 1.
    stddev = 5.
    mean = 0.

    vels = np.arange(-30., 30. + 0.5 * chan_width_stddev * stddev,
                     chan_width_stddev * stddev)
    up_vels = np.arange(-30., 30. + 0.05 * chan_width_stddev * stddev,
                        0.1 * chan_width_stddev * stddev)

    samps = models.Gaussian1D(amplitude=amp,
                              stddev=stddev,
                              mean=mean)(up_vels)

    disc_samps = sample_at_channels(vels, up_vels, samps)

    gauss_samps = gauss_model_discrete(vels, amp, mean, stddev)

    npt.assert_allclose(gauss_samps, disc_samps, atol=5e-3)

    integ = amp * stddev * np.sqrt(2 * np.pi)

    npt.assert_allclose(integ,
                        disc_samps.sum() * chan_width_stddev * stddev,
                        rtol=1e-2)
    npt.assert_allclose(integ,
                        gauss_samps.sum() * chan_width_stddev * stddev,
                        rtol=1e-2)


def test_gauss_discrete():

    integ = gauss_weighted_avg(-100., 100., 1., 5., 0.)

    actual = 1. * 5. * np.sqrt(2 * np.pi)

    npt.assert_allclose(actual, integ * 200.)
