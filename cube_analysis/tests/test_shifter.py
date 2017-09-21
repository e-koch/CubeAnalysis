
import numpy as np
import astropy.units as u
from spectral_cube import SpectralCube
from scipy.optimize import curve_fit
import os

from cube_analysis.spectra_shifter import fourier_shift, cube_shifter
from cube_analysis.tests.utils import generate_hdu


def test_shifter(shape=(100, 100, 100), sigma=8., amp=1.):
    '''
    Use a set of identical Gaussian profiles randomly offset to ensure the
    shifted spectrum has the correct properties.
    '''

    def gaussian(x, amp, mean, sigma):
        return amp * np.exp(- (x - mean)**2 / (2 * sigma**2))

    test_cube = np.empty(shape)
    mean_positions = np.empty(shape[1:])

    spec_middle = shape[0] / 2
    spec_quarter = shape[0] / 4

    np.random.seed(247825498)

    spec_inds = np.mgrid[-spec_middle:spec_middle]
    spat_inds = np.indices(shape[1:])
    for y, x in zip(spat_inds[0].flatten(), spat_inds[1].flatten()):

        # Lock the mean to within 25% from the centre
        mean_pos = np.random.uniform(low=-spec_quarter,
                                     high=spec_quarter)

        mean_positions[y, x] = mean_pos + spec_middle
        test_cube[:, y, x] = gaussian(spec_inds, amp, mean_pos, sigma)

    # Now just use fourier shift to shift all back to the centre
    test_shifted_cube = np.empty(shape)

    for y, x in zip(spat_inds[0].flatten(), spat_inds[1].flatten()):

        pixel_diff = spec_middle - mean_positions[y, x]
        test_shifted_cube[:, y, x] = \
            fourier_shift(test_cube[:, y, x], pixel_diff)

    # Now fit a Gaussian to the mean stacked profile.
    mean_stacked_profile = test_shifted_cube.mean(axis=(1, 2))

    fit_vals = curve_fit(gaussian, spec_inds, mean_stacked_profile)[0]

    np.testing.assert_allclose(fit_vals, np.array([amp, 0.0, sigma]),
                               atol=1e-3)


def test_shifter_wpad(shape=(100, 100, 100), sigma=8., amp=1.):
    '''
    Use a set of identical Gaussian profiles randomly offset to ensure the
    shifted spectrum has the correct properties and add padding before shifting
    so no spectrum wraps around in velocity.
    '''

    def gaussian(x, amp, mean, sigma):
        return amp * np.exp(- (x - mean)**2 / (2 * sigma**2))

    test_cube = np.empty(shape)
    mean_positions = np.empty(shape[1:])

    spec_middle = shape[0] / 2
    spec_quarter = shape[0] / 4

    np.random.seed(247825498)

    spec_inds = np.mgrid[-spec_middle:spec_middle]
    spat_inds = np.indices(shape[1:])
    for y, x in zip(spat_inds[0].flatten(), spat_inds[1].flatten()):

        # Randomly choose the mean to be +/- a quarter of the window
        mean_pos = np.random.choice([1, -1]) * spec_quarter

        mean_positions[y, x] = mean_pos
        test_cube[:, y, x] = gaussian(spec_inds.astype(float), amp,
                                      float(mean_pos), sigma)

    # Convert test_cube into a SpectralCube
    pix_scale = 1 * u.arcsec   # This isn't actually used
    spec_scale = 100 * u.m / u.s
    beamfwhm = 4 * u.arcsec  # This isn't actually used
    test_hdu = generate_hdu(test_cube, pix_scale, spec_scale, beamfwhm)

    test_speccube = SpectralCube.read(test_hdu)

    mean_vels = np.zeros_like(mean_positions) * u.m / u.s
    for y, x in zip(spat_inds[0].flatten(), spat_inds[1].flatten()):
        mean_vels[y, x] = \
            test_speccube.spectral_axis[int(mean_positions[y, x])]

    # Sanity checks on the test data
    assert mean_vels.max() == 3 * spec_quarter * spec_scale
    assert mean_vels.min() == spec_quarter * spec_scale
    assert test_speccube.spectral_extrema[0] == 0.
    # Offset of one channel
    assert test_speccube.spectral_extrema[1] == (shape[0] - 1) * spec_scale

    # Use cube_shifter to do the whole thing, and test if the shape and
    # spectral axis is what should be expected from the shifts
    cube_shifter(test_speccube, mean_vels, save_shifted=True,
                 save_name="test_shift_w_pad.fits",
                 pad_edges=True, return_spectra=False)

    # Open up the saved cube
    save_cube = SpectralCube.read("test_shift_w_pad.fits")

    # The shape of the shifted cube should be v_shape + 2 * v_shape / 4.
    # in this case
    assert save_cube.shape[0] == shape[0] + 2 * spec_quarter

    # The minimum velocity is 3 * spec_quarter (so a half plus a quarter)
    assert save_cube.spectral_extrema[0] == - 3 * spec_quarter * spec_scale
    # Max is this, but positive. Offset of one channel
    assert save_cube.spectral_extrema[1] == (3 * spec_quarter - 1) * spec_scale

    # # Now fit a Gaussian to the mean stacked profile.
    mean_stacked_profile = save_cube.mean(axis=(1, 2))

    fit_vals = curve_fit(gaussian, np.arange(150), mean_stacked_profile,
                         p0=(1, 3 * spec_quarter, 4))[0]

    np.testing.assert_allclose(fit_vals,
                               np.array([amp, 3 * spec_quarter, sigma]),
                               atol=1e-3)

    # Delete the saved cube
    os.system("rm test_shift_w_pad.fits")
