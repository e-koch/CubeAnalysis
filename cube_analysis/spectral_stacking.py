
import numpy as np
import astropy.units as u
from astropy.utils.console import ProgressBar
from astropy.coordinates import Angle


def total_profile(cube, spatial_mask=None, verbose=True):
    '''
    Create the total profile over a region in a given spatial mask.
    '''

    posns = np.where(spatial_mask)

    if verbose:
        spec_iter = ProgressBar(posns[0].size)
    else:
        spec_iter = xrange(posns[0].size)

    total_profile = np.zeros((cube.shape[0],)) * cube.unit

    for i in spec_iter:
        y, x = posns[0][i], posns[1][i]

        spec = cube[:, y, x]
        mask_spec = cube.mask.include(view=(slice(None), y, x))
        valid = np.logical_and(np.isfinite(spec), mask_spec)

        total_profile[valid] += spec[valid]

    return total_profile


def radial_stacking(gal, cube, dr=100 * u.pc, max_radius=8 * u.kpc,
                    pa_bounds=None):
    '''
    Radially stack spectra.
    '''

    max_radius = max_radius.to(dr.unit)

    radius = gal.radius(header=cube.header)

    nbins = np.int(np.floor(max_radius / dr))
    inneredge = np.linspace(0, max_radius - dr, nbins)
    outeredge = np.linspace(dr, max_radius, nbins)

    if pa_bounds is not None:
        # Check if they are angles
        if len(pa_bounds) != 2:
            raise IndexError("pa_bounds must contain 2 angles.")
        if not isinstance(pa_bounds, Angle):
            raise TypeError("pa_bounds must be an Angle.")

        # Return the array of PAs in the galaxy frame
        pas = gal.position_angles(header=cube.header)

        # If the start angle is greater than the end, we need to wrap about
        # the discontinuity
        if pa_bounds[0] > pa_bounds[1]:
            initial_start = pa_bounds[0].copy()
            pa_bounds = pa_bounds.wrap_at(initial_start)
            pas = pas.wrap_at(initial_start)

        pa_mask = np.logical_and(pas >= pa_bounds[0],
                                 pas < pa_bounds[1])
    else:
        pa_mask = np.ones(cube.shape[1:], dtype=bool)

    stacked_spectra = np.zeros((inneredge.size, cube.shape[0])) * cube.unit

    for ctr, (r0, r1) in enumerate(zip(inneredge,
                                       outeredge)):

        print("On bin {} to {}".format(r0.value, r1))

        rad_mask = np.logical_and(radius >= r0, radius < r1)

        spec_mask = np.logical_and(rad_mask, pa_mask)

        stacked_spectra[ctr] = total_profile(cube, spec_mask)

    bin_centers = (inneredge + dr / 2.).to(dr.unit)

    return bin_centers, stacked_spectra
