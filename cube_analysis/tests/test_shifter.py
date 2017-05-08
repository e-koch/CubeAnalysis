
import numpy as np

from cube_analysis.spectra_shifter import fourier_shift


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
    from scipy.optimize import curve_fit

    mean_stacked_profile = test_shifted_cube.mean(axis=(1, 2))

    fit_vals = curve_fit(gaussian, spec_inds, mean_stacked_profile)[0]

    np.testing.assert_allclose(fit_vals, np.array([amp, 0.0, sigma]),
                               atol=1e-3)
