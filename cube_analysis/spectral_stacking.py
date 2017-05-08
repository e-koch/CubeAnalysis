
import numpy as np
import astropy.units as u
from astropy.utils.console import ProgressBar
from astropy.coordinates import Angle
from spectral_cube.cube_utils import _map_context
from itertools import izip


def total_profile(cube, spatial_mask=None, chunk_size=10000,
                  num_cores=1):
    '''
    Create the total profile over a region in a given spatial mask.
    '''

    posns = np.where(spatial_mask)

    posns_y = np.array_split(posns[0], chunk_size)
    posns_x = np.array_split(posns[1], chunk_size)

    cubelist = ([(cube.filled_data[:, jj, ii],
                 cube.mask.include(view=(slice(None), jj, ii)))
                for jj, ii in izip(y_pos, x_pos)]
                for y_pos, x_pos in izip(posns_y, posns_x))

    with _map_context(num_cores) as map:

        stacked_spectra = \
            np.array([x for x in map(_masked_sum, cubelist)])

    # Sum each chunk together
    all_stacked = np.nansum(stacked_spectra, axis=0) * cube.unit

    return all_stacked


def _masked_sum(gen):
    '''
    Sum a list of spectra, applying their respective masks.
    '''

    for i, vals in enumerate(gen):

        spec, mask = vals

        if i == 0:
            total_stack = np.zeros_like(spec)

        total_stack[mask] += spec[mask]

    return total_stack


def radial_stacking(gal, cube, dr=100 * u.pc, max_radius=8 * u.kpc,
                    pa_bounds=None, num_cores=1):
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

        stacked_spectra[ctr] = \
            total_profile(cube, spec_mask, num_cores=num_cores)

    bin_centers = (inneredge + dr / 2.).to(dr.unit)

    return bin_centers, stacked_spectra
